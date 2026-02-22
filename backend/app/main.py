"""
FastAPI Application — Milestone 3
==================================
Production-grade async ticket routing API with:
  • 202 Accepted immediate response
  • Redlock deduplication
  • ARQ-backed background job queue via Redis
  • Pydantic v2 payload validation

Milestone 3 additions:
  • GET  /health         — extended with circuit_breaker_state, agents_online
  • POST /agents         — register a support agent (skill vector)
  • GET  /agents         — list all agents with live load
  • GET  /agents/{id}    — single agent profile
  • POST /agents/{id}/release — release one ticket from agent (decrement load)
  • DELETE /agents/{id}  — unregister an agent
  • GET  /incidents      — list all master flash-flood incidents

Start via:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from fastapi import FastAPI, HTTPException
from urllib.parse import urlparse

from app.config import settings
from app.redis_client import RedlockManager, create_redis_pool
from app.schemas import (
    AcceptedResponse,
    AgentListResponse,
    AgentProfileRequest,
    AgentProfileResponse,
    HealthResponse,
    MasterIncidentResponse,
    TicketPayload,
)

logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — initialise Redis, ARQ, Redlock, and M3 components.
    """
    logger.info("Starting %s v%s …", settings.APP_NAME, settings.APP_VERSION)

    # ── Shared Redis pool ─────────────────────────────────────────────────
    redis = await create_redis_pool()
    app.state.redis   = redis
    app.state.redlock = RedlockManager(redis)

    # ── ARQ connection pool ───────────────────────────────────────────────
    parsed = urlparse(settings.REDIS_URL)
    arq_settings = RedisSettings(
        host=parsed.hostname or "redis",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or "0"),
        username=parsed.username,
        password=parsed.password,
        ssl=True,
        conn_timeout=30,
    )
    app.state.arq_pool: ArqRedis = await create_pool(arq_settings)

    # ── Milestone 3: AgentRegistry (shared with workers via Redis) ────────
    from app.skill_router import AgentRegistry
    app.state.agent_registry = AgentRegistry(redis)

    # ── Milestone 3: SemanticDeduplicator (for incident listing only) ─────
    from app.semantic_dedup import SemanticDeduplicator
    app.state.deduplicator = SemanticDeduplicator(
        redis_client=redis,
        window_secs=settings.FLASH_FLOOD_WINDOW_SECS,
        sim_threshold=settings.FLASH_FLOOD_SIM_THRESHOLD,
        count_threshold=settings.FLASH_FLOOD_COUNT_THRESHOLD,
    )

    # ── Milestone 3: Circuit Breaker state (read-only from API side) ──────
    # The real circuit breaker lives in the worker process; the API reads
    # the stats from a shared Redis key written by the worker if desired.
    # For simplicity, we expose a lightweight placeholder here.
    app.state.circuit_state = "CLOSED"     # default until worker updates it

    logger.info("✓ Server ready — Redis + ARQ + M3 systems initialised.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Shutting down …")
    await app.state.arq_pool.close()
    await app.state.redis.close()
    logger.info("Bye.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Milestone 3 — The Autonomous Orchestrator.\n\n"
        "**Systems**: "
        "SemanticDeduplicator (flash-flood detection) | "
        "CircuitBreaker (latency-aware ML failover) | "
        "SkillBasedRouter (constraint-optimisation agent assignment)"
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Liveness + readiness probe.
    Extended for Milestone 3 with circuit breaker state and agent count.
    """
    try:
        redis_ok     = await app.state.redis.ping()
        redis_status = "connected" if redis_ok else "unreachable"
        if redis_ok:
            cb_state = await app.state.redis.get("circuit_breaker:state")
            if cb_state:
                app.state.circuit_state = cb_state.decode("utf-8")
    except Exception:
        redis_status = "unreachable"

    agents_online = await app.state.agent_registry.count()

    return HealthResponse(
        status         ="ok" if redis_status == "connected" else "degraded",
        redis          =redis_status,
        app_version    =settings.APP_VERSION,
        circuit_breaker_state=app.state.circuit_state,
        agents_online  =agents_online,
    )


# ─── Ticket Submission ────────────────────────────────────────────────────────

@app.post(
    "/tickets",
    response_model=AcceptedResponse,
    status_code=202,
    tags=["Tickets"],
)
async def submit_ticket(payload: TicketPayload):
    """
    Submit a support ticket for async processing.

    **Milestone 3 pipeline** (runs in background ARQ worker):
    1. SemanticDeduplicator → flash-flood detection
    2. CircuitBreaker       → transformer / M1-fallback inference
    3. SkillBasedRouter     → best-agent assignment
    """
    redlock:  RedlockManager = app.state.redlock
    arq_pool: ArqRedis       = app.state.arq_pool

    resource_key = f"{settings.REDLOCK_KEY_PREFIX}{payload.ticket_id}"
    lock_token   = await redlock.acquire(resource=resource_key, ttl_ms=settings.REDLOCK_TTL_MS)

    if lock_token is not None:
        try:
            job_payload: Dict[str, Any] = payload.model_dump()
            await arq_pool.enqueue_job(
                "process_ticket",
                job_payload,
                _queue_name=settings.ARQ_QUEUE_NAME,
            )
            logger.info("Ticket %s enqueued (lock acquired).", payload.ticket_id)
        except Exception as exc:
            await redlock.release(resource_key, lock_token)
            logger.exception("Failed to enqueue ticket %s: %s", payload.ticket_id, exc)
            raise HTTPException(status_code=503, detail=f"Failed to enqueue ticket: {exc}") from exc

        return AcceptedResponse(
            ticket_id=payload.ticket_id,
            status="accepted",
            message="Ticket enqueued for M3 orchestrator pipeline.",
            duplicate=False,
        )

    return AcceptedResponse(
        ticket_id=payload.ticket_id,
        status="accepted",
        message="Ticket already enqueued (duplicate detected via Redlock).",
        duplicate=True,
    )


# ─── Agent Registry ───────────────────────────────────────────────────────────

@app.post("/agents", response_model=AgentProfileResponse, status_code=201, tags=["Agents"])
async def register_agent(req: AgentProfileRequest):
    """
    Register a new support agent with a skill vector.

    Skill values are proficiency scores in [0.0, 1.0]:
    ```json
    { "Billing": 0.9, "Technical": 0.5, "Legal": 0.3 }
    ```
    """
    from app.skill_router import AgentProfile
    profile = AgentProfile(
        agent_id=req.agent_id,
        name=req.name,
        skills=req.skills,
        max_capacity=req.max_capacity,
    )
    await app.state.agent_registry.register(profile)
    return AgentProfileResponse(
        agent_id=profile.agent_id,
        name=profile.name,
        skills=profile.skills,
        max_capacity=profile.max_capacity,
        active_tickets=0,
    )


@app.get("/agents", response_model=AgentListResponse, tags=["Agents"])
async def list_agents():
    """List all registered agents with live load statistics."""
    agents = await app.state.agent_registry.get_all()
    return AgentListResponse(
        agents=[
            AgentProfileResponse(
                agent_id=a.agent_id,
                name=a.name,
                skills=a.skills,
                max_capacity=a.max_capacity,
                active_tickets=a.active_tickets,
            )
            for a in agents
        ],
        total=len(agents),
    )


@app.get("/agents/{agent_id}", response_model=AgentProfileResponse, tags=["Agents"])
async def get_agent(agent_id: str):
    """Retrieve a single agent's profile and current load."""
    agent = await app.state.agent_registry.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    return AgentProfileResponse(
        agent_id=agent.agent_id,
        name=agent.name,
        skills=agent.skills,
        max_capacity=agent.max_capacity,
        active_tickets=agent.active_tickets,
    )


@app.post("/agents/{agent_id}/release", tags=["Agents"])
async def release_agent_ticket(agent_id: str):
    """
    Decrement the active ticket count for an agent (call when a ticket is resolved).
    """
    agent = await app.state.agent_registry.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    new_count = await app.state.agent_registry.decrement_load(agent_id)
    return {"agent_id": agent_id, "active_tickets": new_count}


@app.delete("/agents/{agent_id}", status_code=204, tags=["Agents"])
async def unregister_agent(agent_id: str):
    """Unregister an agent from the routing pool."""
    removed = await app.state.agent_registry.unregister(agent_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")


# ─── Master Incidents ─────────────────────────────────────────────────────────

@app.get("/incidents", tags=["Flash Flood"])
async def list_incidents():
    """
    List all Master Incidents created by the Semantic Deduplicator
    during flash-flood events.
    """
    incidents = await app.state.deduplicator.list_incidents()
    return {"incidents": incidents, "total": len(incidents)}


@app.get("/incidents/{incident_id}", tags=["Flash Flood"])
async def get_incident(incident_id: str):
    """Retrieve a specific Master Incident by ID."""
    incident = await app.state.deduplicator.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")
    return incident
