"""
Orchestrator — Milestone 3
===========================
Top-level glue that wires together all three autonomous systems:

  1. SemanticDeduplicator   → flash-flood detection
  2. CircuitBreaker          → latency-aware ML failover
  3. SkillBasedRouter        → constraint-optimization agent assignment

Call process(ctx, payload) from app/worker.py process_ticket().
Returns an enriched result dict ready for logging / webhook.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates the full Milestone 3 pipeline for a single ticket.

    Parameters
    ----------
    deduplicator : SemanticDeduplicator
    circuit_breaker : CircuitBreaker
    router : SkillBasedRouter
    urgency_threshold : float   — minimum urgency_score to trigger webhook alert
    """

    def __init__(
        self,
        deduplicator,
        circuit_breaker,
        router,
        urgency_threshold: float = 0.8,
    ):
        self.deduplicator      = deduplicator
        self.circuit_breaker   = circuit_breaker
        self.router            = router
        self.urgency_threshold = urgency_threshold

    async def process(
        self,
        ctx:     Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Full Milestone 3 processing pipeline.

        Returns a result dict with all fields needed by the worker to:
          • decide whether to fire a webhook
          • log structured output
          • respond to status polls
        """
        ticket_id = payload.get("ticket_id", "unknown")
        subject   = payload.get("subject", "")
        body      = payload.get("body", "")
        full_text = f"{subject} {body}".strip()

        result: Dict[str, Any] = {
            "ticket_id":          ticket_id,
            "status":             "processed",
            # --- filled in below ---
            "category":           None,
            "confidence":         None,
            "urgency_score":      None,
            "alert_sent":         False,
            "is_flood":           False,
            "similar_count":      0,
            "master_incident_id": None,
            "circuit_state":      None,
            "routed_to":          None,
            "routing_score":      None,
            "routing_reason":     None,
            "model_used":         None,
        }

        # ── Step 1: Semantic Deduplication ────────────────────────────────────
        try:
            dedup = await self.deduplicator.check(ticket_id, full_text)
            result["is_flood"]           = dedup.is_flood
            result["similar_count"]      = dedup.similar_count
            result["master_incident_id"] = dedup.master_incident_id

            if dedup.is_flood:
                logger.warning(
                    "[Orchestrator] ⚡ Flash Flood! ticket=%s  incident=%s  similar=%d",
                    ticket_id, dedup.master_incident_id, dedup.similar_count,
                )
        except Exception as exc:
            logger.error("[Orchestrator] Deduplicator error for %s: %s", ticket_id, exc)

        # ── Step 2: ML Inference via Circuit Breaker ──────────────────────────
        try:
            category, confidence, urgency_score = await self.circuit_breaker.call(
                ctx, full_text
            )
            cb_state = self.circuit_breaker.state.value

            result["category"]      = category
            result["confidence"]    = round(confidence, 4)
            result["urgency_score"] = round(urgency_score, 4)
            result["circuit_state"] = cb_state
            # Track which model was actually used
            result["model_used"] = (
                "transformer"
                if cb_state == "CLOSED" or cb_state == "HALF_OPEN"
                else "ensemble_ir_fallback"
            )

            logger.info(
                "[Orchestrator] Inference OK  ticket=%s  category=%s  "
                "urgency=%.4f  circuit=%s  model=%s",
                ticket_id, category, urgency_score, cb_state, result["model_used"],
            )
        except Exception as exc:
            logger.exception("[Orchestrator] CircuitBreaker error for %s: %s", ticket_id, exc)
            result["status"] = "error"
            result["error"]  = str(exc)
            return result

        # ── Step 3: Skill-Based Routing ───────────────────────────────────────
        try:
            urgency_label = _score_to_label(urgency_score)
            decision = await self.router.route(category, urgency_label)
            result["routed_to"]      = decision.assigned_agent_id
            result["routing_score"]  = decision.score
            result["routing_reason"] = decision.reason

            if decision.assigned_agent_id:
                logger.info(
                    "[Orchestrator] Routed ticket=%s → agent=%s (score=%.4f)",
                    ticket_id, decision.assigned_agent_id, decision.score,
                )
            else:
                logger.warning(
                    "[Orchestrator] No agent available for ticket=%s: %s",
                    ticket_id, decision.reason,
                )
        except Exception as exc:
            logger.error("[Orchestrator] Router error for %s: %s", ticket_id, exc)

        # ── Step 4: Webhook gate ──────────────────────────────────────────────
        # Suppress individual alerts during a flood; only the Master Incident fires.
        should_alert = (
            urgency_score > self.urgency_threshold
            and not result["is_flood"]
        )
        result["should_alert"] = should_alert

        return result


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _score_to_label(urgency_score: float) -> str:
    """Map urgency regression score to HIGH / MEDIUM / LOW label."""
    if urgency_score >= 0.75:
        return "HIGH"
    if urgency_score >= 0.30:
        return "MEDIUM"
    return "LOW"
