"""
ARQ Worker — Milestone 3
=========================
Background worker process that consumes ticket jobs from Redis via ARQ.

For each ticket the Milestone 3 Orchestrator pipeline runs:
  1. SemanticDeduplicator  → flash-flood detection via sentence embeddings
  2. CircuitBreaker         → latency-aware failover (Transformer → M1 Ensemble)
  3. SkillBasedRouter       → constraint-optimization agent assignment
  4. (Optional) Webhook     → alert on urgent, non-flood tickets

Start via:  arq app.worker.WorkerSettings
"""

import logging
from typing import Any, Dict
from urllib.parse import urlparse

from arq.connections import RedisSettings

from app.config import settings

logger = logging.getLogger(__name__)


# ─── Startup / Shutdown Hooks ─────────────────────────────────────────────────

async def startup(ctx: Dict[str, Any]) -> None:
    """
    Called once when the worker process starts.
    Initialises all Milestone 3 components and shares them via ctx.
    """
    from multitask_model import MultiTaskPredictor
    from classifier import EnsembleIRClassifier
    from app.semantic_dedup import SemanticDeduplicator
    from app.circuit_breaker import CircuitBreaker
    from app.skill_router import AgentRegistry, SkillBasedRouter
    from app.orchestrator import Orchestrator
    from app.redis_client import create_redis_pool

    logger.info("Worker starting — loading models …")

    # ── Primary transformer model (Milestone 2) ───────────────────────────
    ctx["predictor"] = MultiTaskPredictor()
    logger.info("Worker ✓ MultiTaskPredictor loaded on %s", ctx["predictor"].device)

    # ── Fallback model (Milestone 1) ──────────────────────────────────────
    fallback = EnsembleIRClassifier()
    fallback.load_or_train()
    ctx["fallback_classifier"] = fallback
    logger.info("Worker ✓ EnsembleIRClassifier (M1 fallback) loaded.")

    # ── Redis client for M3 subsystems ────────────────────────────────────
    redis = await create_redis_pool()
    ctx["m3_redis"] = redis

    # ── Semantic Deduplicator ─────────────────────────────────────────────
    ctx["deduplicator"] = SemanticDeduplicator(
        redis_client=redis,
        window_secs=settings.FLASH_FLOOD_WINDOW_SECS,
        sim_threshold=settings.FLASH_FLOOD_SIM_THRESHOLD,
        count_threshold=settings.FLASH_FLOOD_COUNT_THRESHOLD,
    )
    logger.info("Worker ✓ SemanticDeduplicator ready (window=%ds).",
                settings.FLASH_FLOOD_WINDOW_SECS)

    # ── Circuit Breaker ───────────────────────────────────────────────────
    ctx["circuit_breaker"] = CircuitBreaker(
        latency_threshold_ms=settings.CIRCUIT_BREAKER_LATENCY_MS,
        cooldown_secs=settings.CIRCUIT_BREAKER_COOLDOWN_SECS,
    )
    logger.info("Worker ✓ CircuitBreaker ready (threshold=%dms).",
                settings.CIRCUIT_BREAKER_LATENCY_MS)

    # ── Skill-Based Router ────────────────────────────────────────────────
    registry = AgentRegistry(redis)
    ctx["agent_registry"] = registry
    ctx["router"]         = SkillBasedRouter(registry)
    logger.info("Worker ✓ SkillBasedRouter ready.")

    # ── Orchestrator ──────────────────────────────────────────────────────
    ctx["orchestrator"] = Orchestrator(
        deduplicator=ctx["deduplicator"],
        circuit_breaker=ctx["circuit_breaker"],
        router=ctx["router"],
        urgency_threshold=settings.URGENCY_THRESHOLD,
    )
    logger.info("Worker ✓ Orchestrator ready — all M3 systems online.\n")


async def shutdown(ctx: Dict[str, Any]) -> None:
    """Called once when the worker process shuts down."""
    redis = ctx.get("m3_redis")
    if redis:
        await redis.close()
    logger.info("Worker shut down.")


# ─── Job: Process Ticket ──────────────────────────────────────────────────────

async def process_ticket(ctx: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    ARQ job function — full Milestone 3 pipeline for a single ticket.

    Args:
        ctx:     Worker context (predictor, fallback_classifier, orchestrator, …)
        payload: Ticket data dict: ticket_id, subject, body, customer_id, channel.

    Returns:
        Enriched result dict with category, confidence, urgency_score,
        circuit_state, is_flood, master_incident_id, routed_to, alert_sent.
    """
    from app.webhook import send_webhook_alert

    ticket_id = payload.get("ticket_id", "unknown")
    logger.info("Processing ticket %s …", ticket_id)

    orchestrator = ctx.get("orchestrator")

    # ── If orchestrator not available (e.g. test env), fall back to M2 ────
    if orchestrator is None:
        logger.warning("Orchestrator not in ctx — running M2 path.")
        return await _process_ticket_m2_fallback(ctx, payload)

    # ── Full M3 Orchestrator pipeline ─────────────────────────────────────
    result = await orchestrator.process(ctx, payload)

    # ── Webhook alert (suppressed during flash-flood) ─────────────────────
    if result.get("should_alert"):
        alert_data = {**payload, **{
            "category":      result.get("category"),
            "confidence":    result.get("confidence"),
            "urgency_score": result.get("urgency_score"),
            "routed_to":     result.get("routed_to"),
        }}
        alert_sent = await send_webhook_alert(alert_data)
        result["alert_sent"] = alert_sent
    elif result.get("is_flood"):
        logger.info(
            "Ticket %s belongs to flood incident %s — individual alert suppressed.",
            ticket_id, result.get("master_incident_id"),
        )

    # Sync state to Redis for the API to read
    redis = ctx.get("m3_redis")
    if redis:
        try:
            await redis.set("circuit_breaker:state", result.get("circuit_state", "CLOSED"))
        except Exception as e:
            logger.error(f"Failed to sync CB state to Redis: {e}")

    logger.info(
        "Ticket %s done — category=%s  circuit=%s  flood=%s  agent=%s",
        ticket_id,
        result.get("category"),
        result.get("circuit_state"),
        result.get("is_flood"),
        result.get("routed_to"),
    )
    return result


# ─── M2 Fallback (no orchestrator context) ────────────────────────────────────

async def _process_ticket_m2_fallback(
    ctx: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Reproduce Milestone 2 behaviour as a safe fallback."""
    from app.webhook import send_webhook_alert

    predictor = ctx["predictor"]
    ticket_id = payload.get("ticket_id", "unknown")
    full_text  = f"{payload.get('subject', '')} {payload.get('body', '')}".strip()

    try:
        category, confidence, urgency_score = predictor.predict(full_text)
    except Exception as exc:
        return {"ticket_id": ticket_id, "status": "error", "error": str(exc)}

    result = {
        "ticket_id":     ticket_id,
        "category":      category,
        "confidence":    round(confidence, 4),
        "urgency_score": round(urgency_score, 4),
        "status":        "processed",
        "alert_sent":    False,
        "model_used":    "transformer",
    }

    if urgency_score > settings.URGENCY_THRESHOLD:
        alert_sent = await send_webhook_alert({**payload, **result})
        result["alert_sent"] = alert_sent

    return result


# ─── ARQ Worker Settings ─────────────────────────────────────────────────────

def _parse_redis_settings() -> RedisSettings:
    """Parse REDIS_URL into ARQ RedisSettings, supporting TLS."""
    url    = settings.REDIS_URL
    parsed = urlparse(url)
    is_tls = url.startswith("rediss://")
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or "0"),
        username=parsed.username,
        password=parsed.password,
        ssl=is_tls,
    )


class WorkerSettings:
    """ARQ worker configuration — discovered by `arq app.worker.WorkerSettings`."""

    functions          = [process_ticket]
    on_startup         = startup
    on_shutdown        = shutdown
    redis_settings     = _parse_redis_settings()
    queue_name         = settings.ARQ_QUEUE_NAME
    max_jobs           = 10
    job_timeout        = 180           # seconds (longer for M3 — embeddings are heavier)
    keep_result        = 3600
    health_check_interval = 30
