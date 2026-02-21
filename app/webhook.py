"""
Async Webhook Alerting — Milestone 2
=====================================
Sends high-urgency ticket alerts to a mock Discord/Slack webhook
when the regression head predicts S > 0.8.

Features:
  • Fully async via httpx.AsyncClient
  • Retry with exponential backoff (3 attempts)
  • Graceful error handling — never crashes the worker
"""

import logging
from typing import Any, Dict

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Retry Configuration ─────────────────────────────────────────────────────

MAX_RETRIES: int = 3
BASE_DELAY_SECONDS: float = 0.5   # 0.5s → 1.0s → 2.0s


async def send_webhook_alert(ticket_data: Dict[str, Any]) -> bool:
    """
    POST an urgent ticket alert to the configured webhook URL.

    Payload format mimics Discord/Slack webhook conventions:
        {
            "content": "🚨 URGENT TICKET ALERT",
            "embeds": [{ ... ticket details ... }]
        }

    Args:
        ticket_data: Dict with ticket_id, category, urgency_score, subject, body, etc.

    Returns:
        True if the webhook was delivered successfully, False otherwise.
    """
    payload = {
        "content": "🚨 **URGENT TICKET ALERT** — Urgency score exceeded threshold",
        "embeds": [
            {
                "title": f"Ticket: {ticket_data.get('ticket_id', 'N/A')}",
                "color": 15158332,  # Red
                "fields": [
                    {
                        "name": "Category",
                        "value": ticket_data.get("category", "Unknown"),
                        "inline": True,
                    },
                    {
                        "name": "Urgency Score",
                        "value": f"{ticket_data.get('urgency_score', 0.0):.4f}",
                        "inline": True,
                    },
                    {
                        "name": "Confidence",
                        "value": f"{ticket_data.get('confidence', 0.0):.4f}",
                        "inline": True,
                    },
                    {
                        "name": "Subject",
                        "value": ticket_data.get("subject", "—")[:256],
                        "inline": False,
                    },
                    {
                        "name": "Customer",
                        "value": ticket_data.get("customer_id", "N/A"),
                        "inline": True,
                    },
                    {
                        "name": "Channel",
                        "value": ticket_data.get("channel", "N/A"),
                        "inline": True,
                    },
                ],
            }
        ],
    }

    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    settings.WEBHOOK_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

            logger.info(
                "Webhook delivered for ticket %s (attempt %d/%d, status=%d)",
                ticket_data.get("ticket_id"),
                attempt,
                MAX_RETRIES,
                response.status_code,
            )
            return True

        except httpx.HTTPStatusError as exc:
            last_exc = exc
            logger.warning(
                "Webhook HTTP error for ticket %s: %d %s (attempt %d/%d)",
                ticket_data.get("ticket_id"),
                exc.response.status_code,
                exc.response.reason_phrase,
                attempt,
                MAX_RETRIES,
            )
        except httpx.RequestError as exc:
            last_exc = exc
            logger.warning(
                "Webhook connection error for ticket %s: %s (attempt %d/%d)",
                ticket_data.get("ticket_id"),
                str(exc),
                attempt,
                MAX_RETRIES,
            )

        # Exponential backoff before next retry
        if attempt < MAX_RETRIES:
            import asyncio
            delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            await asyncio.sleep(delay)

    logger.error(
        "Webhook FAILED after %d attempts for ticket %s: %s",
        MAX_RETRIES,
        ticket_data.get("ticket_id"),
        str(last_exc),
    )
    return False
