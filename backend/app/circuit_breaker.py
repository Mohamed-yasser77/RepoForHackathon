"""
CircuitBreaker — Milestone 3
==============================
Wraps calls to the MultiTaskPredictor (Transformer / MiniLM) with a three-state
circuit breaker that monitors inference latency and automatically fails over to
the lightweight Milestone 1 EnsembleIRClassifier when the Transformer is too slow.

States
------
  CLOSED    — Transformer is the active model.  All calls go to it.
              Tracks a rolling Exponential Weighted Moving Average (EWMA) of
              inference latency.  If EWMA > LATENCY_THRESHOLD_MS the circuit trips.

  OPEN      — Transformer is bypassed.  ALL calls go to the fallback (M1 model).
              After COOLDOWN_SECS the circuit transitions to HALF_OPEN.

  HALF_OPEN — One probe request is sent to the Transformer.
              • If latency ≤ threshold → circuit CLOSES again.
              • If latency > threshold  → circuit stays/transitions back to OPEN.

Thread-safety
-------------
All state mutations are protected by a threading.Lock so the CircuitBreaker
can be safely shared across concurrent ARQ worker coroutines (which run in an
asyncio event loop on a single thread, but we guard against future changes).
"""

import time
import logging
import threading
from enum import Enum
from typing import Tuple, Optional, Any

logger = logging.getLogger(__name__)


# ─── State Enum ───────────────────────────────────────────────────────────────

class CBState(str, Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# ─── CircuitBreaker ───────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Latency-aware circuit breaker for ML model calls.

    Usage (inside async process_ticket):
        result = await circuit_breaker.call(ctx, full_text)
        # result is (category, confidence, urgency_score)
    """

    def __init__(
        self,
        latency_threshold_ms: int   = 500,
        cooldown_secs:        int   = 30,
        ewma_alpha:           float = 0.3,   # smoothing factor
    ):
        self.latency_threshold_ms = latency_threshold_ms
        self.cooldown_secs        = cooldown_secs
        self.alpha                = ewma_alpha        # EWMA α ∈ (0, 1]

        self._state:       CBState   = CBState.CLOSED
        self._ewma_ms:     float     = 0.0
        self._opened_at:   Optional[float] = None    # UNIX timestamp when opened
        self._probe_sent:  bool      = False
        self._lock:        threading.Lock = threading.Lock()

        # Stats
        self._total_calls:         int = 0
        self._transformer_calls:   int = 0
        self._fallback_calls:      int = 0
        self._trips:               int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> CBState:
        with self._lock:
            return self._state

    @property
    def ewma_latency_ms(self) -> float:
        with self._lock:
            return round(self._ewma_ms, 2)

    def stats(self) -> dict:
        with self._lock:
            return {
                "state":              self._state.value,
                "ewma_latency_ms":    round(self._ewma_ms, 2),
                "total_calls":        self._total_calls,
                "transformer_calls":  self._transformer_calls,
                "fallback_calls":     self._fallback_calls,
                "circuit_trips":      self._trips,
                "cooldown_remaining": self._cooldown_remaining(),
            }

    def _cooldown_remaining(self) -> float:
        """Seconds until OPEN → HALF_OPEN transition (0 if already elapsed)."""
        if self._state != CBState.OPEN or self._opened_at is None:
            return 0.0
        elapsed = time.time() - self._opened_at
        return max(0.0, self.cooldown_secs - elapsed)

    # ── Core call ─────────────────────────────────────────────────────────────

    async def call(
        self,
        ctx: dict,
        full_text: str,
    ) -> Tuple[str, float, float]:
        """
        Route inference to the appropriate model based on circuit state.

        Returns:
            (category: str, confidence: float, urgency_score: float)
        """
        with self._lock:
            self._total_calls += 1
            current_state = self._state
            # Check if OPEN cooldown has elapsed → transition to HALF_OPEN
            if current_state == CBState.OPEN and self._opened_at is not None:
                if time.time() - self._opened_at >= self.cooldown_secs:
                    self._state    = CBState.HALF_OPEN
                    self._probe_sent = False
                    current_state  = CBState.HALF_OPEN
                    logger.info("[CB] OPEN → HALF_OPEN after cooldown.")

        if current_state == CBState.CLOSED:
            return await self._call_transformer(ctx, full_text)

        elif current_state == CBState.OPEN:
            return self._call_fallback(ctx, full_text, reason="circuit OPEN")

        else:   # HALF_OPEN
            with self._lock:
                if self._probe_sent:
                    # probe already in-flight; use fallback for parallel requests
                    return self._call_fallback(ctx, full_text, reason="probe in-flight")
                self._probe_sent = True

            # Send probe to transformer
            return await self._probe_transformer(ctx, full_text)

    # ── Transformer path ──────────────────────────────────────────────────────

    async def _call_transformer(
        self, ctx: dict, full_text: str
    ) -> Tuple[str, float, float]:
        predictor = ctx["predictor"]
        t0 = time.perf_counter()
        category, confidence, urgency_score = predictor.predict(full_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        with self._lock:
            self._transformer_calls += 1
            # Update EWMA
            if self._ewma_ms == 0.0:
                self._ewma_ms = elapsed_ms
            else:
                self._ewma_ms = self.alpha * elapsed_ms + (1 - self.alpha) * self._ewma_ms

            logger.debug(
                "[CB] Transformer OK  %.1f ms  EWMA=%.1f ms",
                elapsed_ms, self._ewma_ms,
            )

            # Trip check
            if self._ewma_ms > self.latency_threshold_ms:
                self._state     = CBState.OPEN
                self._opened_at = time.time()
                self._trips    += 1
                logger.warning(
                    "[CB] CLOSED → OPEN  EWMA=%.1f ms > threshold=%d ms  (trip #%d)",
                    self._ewma_ms, self.latency_threshold_ms, self._trips,
                )

        return category, confidence, urgency_score

    async def _probe_transformer(
        self, ctx: dict, full_text: str
    ) -> Tuple[str, float, float]:
        predictor = ctx["predictor"]
        t0 = time.perf_counter()
        try:
            category, confidence, urgency_score = predictor.predict(full_text)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            with self._lock:
                self._transformer_calls += 1
                self._ewma_ms = elapsed_ms   # reset EWMA on probe

                if elapsed_ms <= self.latency_threshold_ms:
                    self._state = CBState.CLOSED
                    logger.info(
                        "[CB] HALF_OPEN → CLOSED  probe latency=%.1f ms ✓",
                        elapsed_ms,
                    )
                else:
                    self._state     = CBState.OPEN
                    self._opened_at = time.time()
                    self._trips    += 1
                    logger.warning(
                        "[CB] HALF_OPEN → OPEN  probe latency=%.1f ms still too high.",
                        elapsed_ms,
                    )

            return category, confidence, urgency_score

        except Exception as exc:
            with self._lock:
                self._state     = CBState.OPEN
                self._opened_at = time.time()
                self._trips    += 1
                logger.error("[CB] Probe failed (%s) → OPEN", exc)
            return self._call_fallback(ctx, full_text, reason="probe exception")

    # ── Fallback path ─────────────────────────────────────────────────────────

    def _call_fallback(
        self, ctx: dict, full_text: str, reason: str
    ) -> Tuple[str, float, float]:
        """Use the M1 EnsembleIRClassifier as a lightweight fallback."""
        fallback = ctx.get("fallback_classifier")
        if fallback is None:
            logger.error("[CB] Fallback classifier not in ctx — returning defaults.")
            return "Technical", 0.5, 0.5

        with self._lock:
            self._fallback_calls += 1

        logger.info("[CB] Using M1 fallback (%s).", reason)
        try:
            category, confidence, votes = fallback.predict(full_text)
            # EnsembleIRClassifier does not produce urgency_score — derive from confidence
            urgency_score = min(0.95, confidence)
            return category, confidence, urgency_score
        except Exception as exc:
            logger.error("[CB] Fallback also failed: %s", exc)
            return "Technical", 0.5, 0.5
