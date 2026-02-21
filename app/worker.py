"""
ARQ Worker — Milestone 2
=========================
Background worker process that consumes ticket jobs from Redis via ARQ.

For each ticket:
  1. Runs MultiTaskPredictor.predict() → (category, confidence, urgency_score)
  2. If urgency_score > threshold → triggers async webhook alert
  3. Logs structured results

Start via:  arq app.worker.WorkerSettings
"""

import logging
from typing import Any, Dict

from arq import cron
from arq.connections import RedisSettings

from app.config import settings

logger = logging.getLogger(__name__)


# ─── Startup / Shutdown Hooks ─────────────────────────────────────────────────

async def startup(ctx: Dict[str, Any]) -> None:
    """
    Called once when the worker process starts.
    Initialises the ML model so it is shared across all job executions.
    """
    from multitask_model import MultiTaskPredictor

    logger.info("Worker starting — loading MultiTaskPredictor …")
    ctx["predictor"] = MultiTaskPredictor()
    logger.info("Worker ready — model loaded on %s", ctx["predictor"].device)


async def shutdown(ctx: Dict[str, Any]) -> None:
    """Called once when the worker process shuts down."""
    logger.info("Worker shutting down.")


# ─── Job: Process Ticket ──────────────────────────────────────────────────────

async def process_ticket(ctx: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    ARQ job function — runs inference on a ticket and optionally fires a webhook.

    Args:
        ctx:     Worker context containing the shared predictor instance.
        payload: Ticket data dict with keys: ticket_id, subject, body,
                 customer_id, channel.

    Returns:
        Result dict with category, confidence, urgency_score, and alert status.
    """
    from multitask_model import MultiTaskPredictor
    from app.webhook import send_webhook_alert

    predictor: MultiTaskPredictor = ctx["predictor"]

    ticket_id = payload.get("ticket_id", "unknown")
    subject = payload.get("subject", "")
    body = payload.get("body", "")
    full_text = f"{subject} {body}".strip()

    logger.info("Processing ticket %s …", ticket_id)

    # ── 1. ML Inference ───────────────────────────────────────────────────
    try:
        category, confidence, urgency_score = predictor.predict(full_text)
    except Exception as exc:
        logger.exception("Inference failed for ticket %s: %s", ticket_id, exc)
        return {
            "ticket_id": ticket_id,
            "status": "error",
            "error": str(exc),
        }

    result = {
        "ticket_id": ticket_id,
        "category": category,
        "confidence": round(confidence, 4),
        "urgency_score": round(urgency_score, 4),
        "status": "processed",
        "alert_sent": False,
    }

    logger.info(
        "Ticket %s → category=%s, confidence=%.4f, urgency=%.4f",
        ticket_id, category, confidence, urgency_score,
    )

    # ── 2. Webhook Alert (if urgency > threshold) ────────────────────────
    if urgency_score > settings.URGENCY_THRESHOLD:
        logger.info(
            "Urgency %.4f exceeds threshold %.2f — sending webhook alert for %s",
            urgency_score, settings.URGENCY_THRESHOLD, ticket_id,
        )
        alert_data = {
            **payload,
            "category": category,
            "confidence": round(confidence, 4),
            "urgency_score": round(urgency_score, 4),
        }
        alert_sent = await send_webhook_alert(alert_data)
        result["alert_sent"] = alert_sent

    return result


# ─── ARQ Worker Settings ─────────────────────────────────────────────────────

def _parse_redis_settings() -> RedisSettings:
    """Parse the REDIS_URL into ARQ's RedisSettings, supporting TLS."""
    from urllib.parse import urlparse

    url = settings.REDIS_URL
    parsed = urlparse(url)
    is_tls = url.startswith("rediss://")

    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or "0"),
        password=parsed.password,
        ssl=is_tls,
    )


class WorkerSettings:
    """ARQ worker configuration — discovered automatically by `arq app.worker.WorkerSettings`."""

    functions = [process_ticket]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _parse_redis_settings()
    queue_name = settings.ARQ_QUEUE_NAME
    max_jobs = 10
    job_timeout = 120          # seconds
    keep_result = 3600         # keep results for 1 hour
    health_check_interval = 30
