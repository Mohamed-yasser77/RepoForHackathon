"""
FastAPI Application — Milestone 2
==================================
Production-grade async ticket routing API with:
  • 202 Accepted immediate response
  • Redlock deduplication (10 identical ticket_ids → only 1 enqueued)
  • ARQ-backed background job queue via Redis
  • Pydantic v2 payload validation

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
from app.schemas import AcceptedResponse, HealthResponse, TicketPayload

logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — initialise Redis, ARQ pool, and Redlock on startup;
    tear down on shutdown.
    """
    # ── Startup ───────────────────────────────────────────────────────────
    logger.info("Starting %s v%s …", settings.APP_NAME, settings.APP_VERSION)

    # Redis connection pool (for Redlock)
    redis = await create_redis_pool()
    app.state.redis = redis
    app.state.redlock = RedlockManager(redis)

    # ARQ connection pool (for job enqueuing)
    parsed = urlparse(settings.REDIS_URL)
    # ✅ FIXED — add ssl and username
    arq_redis_settings = RedisSettings(
        host=parsed.hostname or "redis",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or "0"),
        username=parsed.username,   # ← "default" from your URL
        password=parsed.password,
        ssl=True,                   # ← this is the only thing that was missing
        conn_timeout=30,            # ← Upstash cold starts can be slow
    )
    app.state.arq_pool: ArqRedis = await create_pool(arq_redis_settings)

    logger.info("✓ Server ready — Redis + ARQ initialised.")
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
        "Milestone 2 — Production-grade async ticket routing engine.\n\n"
        "**Architecture**: FastAPI → Redlock dedup → ARQ (Redis) → "
        "Multi-Task MiniLM (classification + urgency) → Webhook alerts"
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Liveness + readiness probe.
    Pings Redis to confirm the backing store is reachable.
    """
    try:
        redis_ok = await app.state.redis.ping()
        redis_status = "connected" if redis_ok else "unreachable"
    except Exception:
        redis_status = "unreachable"

    return HealthResponse(
        status="ok" if redis_status == "connected" else "degraded",
        redis=redis_status,
        app_version=settings.APP_VERSION,
    )


@app.post(
    "/tickets",
    response_model=AcceptedResponse,
    status_code=202,
    tags=["Tickets"],
)
async def submit_ticket(payload: TicketPayload):
    """
    Submit a support ticket for async processing.

    **Behaviour**:
    1. Validates the payload via Pydantic.
    2. Attempts Redlock on `ticket_lock:{ticket_id}` (TTL = REDLOCK_TTL_MS).
    3. If lock acquired → enqueues a background job to the ARQ worker.
    4. If lock NOT acquired → this is a duplicate; still returns 202.
    5. ALL requests return 202 Accepted immediately — no blocking.

    **Deduplication guarantee**: If 10 identical ticket_ids arrive
    in the same millisecond, exactly one ARQ job is created.
    """
    redlock: RedlockManager = app.state.redlock
    arq_pool: ArqRedis = app.state.arq_pool

    resource_key = f"{settings.REDLOCK_KEY_PREFIX}{payload.ticket_id}"

    # ── Attempt distributed lock ──────────────────────────────────────────
    lock_token = await redlock.acquire(
        resource=resource_key,
        ttl_ms=settings.REDLOCK_TTL_MS,
    )

    if lock_token is not None:
        # ── Lock acquired — first submission of this ticket_id ────────────
        try:
            job_payload: Dict[str, Any] = payload.model_dump()
            await arq_pool.enqueue_job(
                "process_ticket",
                job_payload,
                _queue_name=settings.ARQ_QUEUE_NAME,
            )
            logger.info(
                "Ticket %s enqueued to ARQ (lock acquired).", payload.ticket_id
            )
        except Exception as exc:
            # Release lock on enqueue failure so a retry can succeed
            await redlock.release(resource_key, lock_token)
            logger.exception("Failed to enqueue ticket %s: %s", payload.ticket_id, exc)
            raise HTTPException(
                status_code=503,
                detail=f"Failed to enqueue ticket: {exc}",
            ) from exc

        return AcceptedResponse(
            ticket_id=payload.ticket_id,
            status="accepted",
            message="Ticket enqueued for background processing.",
            duplicate=False,
        )

    else:
        # ── Lock NOT acquired — duplicate ticket_id ───────────────────────
        logger.info(
            "Ticket %s is a duplicate (Redlock denied). Returning 202 anyway.",
            payload.ticket_id,
        )
        return AcceptedResponse(
            ticket_id=payload.ticket_id,
            status="accepted",
            message="Ticket already enqueued (duplicate detected via Redlock).",
            duplicate=True,
        )
