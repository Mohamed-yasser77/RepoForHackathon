"""
SemanticDeduplicator — Milestone 3
====================================
Flash-Flood detection via sentence embeddings + Redis rolling window.

Algorithm:
  1. Encode incoming ticket text with all-MiniLM-L6-v2 (384-dim, L2-normalised).
  2. Store embedding in a Redis ZSET keyed by UNIX timestamp (score).
  3. At query time:
       • Retrieve all embeddings within the last WINDOW_SECS seconds.
       • Compute cosine similarity against each (dot product on normalised vecs).
       • Count how many have similarity > SIM_THRESHOLD.
  4. If similar_count > COUNT_THRESHOLD → Flash-Flood declared.
       • Suppress individual webhook alerts.
       • Create or link to a "Master Incident" stored as a Redis Hash.
  5. Returns NamedTuple: (is_flood, master_incident_id, similar_count, embedding).

Redis key layout:
  dedup:embeddings   → ZSET  { score=timestamp, member=json{ticket_id, embedding} }
  dedup:incidents    → HASH  { incident_id → json blob }
  dedup:active       → STRING (current active master incident id, or empty)
"""

import json
import time
import uuid
import logging
import threading
from typing import Optional, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

DEDUP_EMBEDDINGS_KEY = "dedup:embeddings"
DEDUP_INCIDENTS_KEY  = "dedup:incidents"
DEDUP_ACTIVE_KEY     = "dedup:active_incident"

_model_lock = threading.Lock()
_model      = None          # lazy-loaded sentence-transformers model


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                logger.info("[SemanticDedup] Loading all-MiniLM-L6-v2 …")
                _model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("[SemanticDedup] ✓ Model loaded.")
    return _model


# ─── Return type ──────────────────────────────────────────────────────────────

class DedupResult(NamedTuple):
    is_flood:           bool
    master_incident_id: Optional[str]
    similar_count:      int
    embedding:          list        # raw list for JSON serialisation


# ─── Main Class ───────────────────────────────────────────────────────────────

class SemanticDeduplicator:
    """
    Stateless per-request; all state lives in Redis.
    Instantiate once and share across worker context.
    """

    def __init__(
        self,
        redis_client,
        window_secs:      int   = 300,
        sim_threshold:    float = 0.9,
        count_threshold:  int   = 10,
    ):
        self.redis           = redis_client
        self.window_secs     = window_secs
        self.sim_threshold   = sim_threshold
        self.count_threshold = count_threshold

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Return an L2-normalised 384-dim embedding vector."""
        model = _get_model()
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    # ── Core check ────────────────────────────────────────────────────────────

    async def check(self, ticket_id: str, text: str) -> DedupResult:
        """
        Main entry point.  Call this before ML inference.

        Returns DedupResult with is_flood=True if a flash flood is detected.
        Always stores the current embedding in Redis (for future checks).
        """
        now   = time.time()
        start = now - self.window_secs

        # 1. Encode current ticket
        embedding = self.encode(text)

        # 2. Retrieve recent embeddings from Redis ZSET
        raw_members = await self.redis.zrangebyscore(
            DEDUP_EMBEDDINGS_KEY, start, now
        )

        # 3. Compute cosine similarities
        similar_count = 0
        for raw in raw_members:
            try:
                record = json.loads(raw)
                other_vec = np.array(record["embedding"], dtype=np.float32)
                sim = float(np.dot(embedding, other_vec))   # both L2-normalised → dot = cosine
                if sim >= self.sim_threshold:
                    similar_count += 1
            except Exception:
                pass    # corrupt record — skip

        # 4. Store current embedding
        member = json.dumps({
            "ticket_id": ticket_id,
            "embedding": embedding.tolist(),
        })
        await self.redis.zadd(DEDUP_EMBEDDINGS_KEY, {member: now})
        # Expire old entries (keep only last 2× window to be safe)
        await self.redis.zremrangebyscore(DEDUP_EMBEDDINGS_KEY, 0, now - self.window_secs * 2)
        # TTL on the whole key so Redis auto-cleans
        await self.redis.expire(DEDUP_EMBEDDINGS_KEY, self.window_secs * 4)

        # 5. Flash-flood decision
        is_flood = similar_count >= self.count_threshold
        master_incident_id: Optional[str] = None

        if is_flood:
            master_incident_id = await self._get_or_create_incident(
                similar_count=similar_count,
                sample_text=text,
            )
            logger.warning(
                "[SemanticDedup] ⚡ FLASH FLOOD detected! "
                "similar_count=%d  incident=%s",
                similar_count, master_incident_id,
            )
        else:
            logger.debug(
                "[SemanticDedup] No flood. similar_count=%d (threshold=%d)",
                similar_count, self.count_threshold,
            )

        return DedupResult(
            is_flood=is_flood,
            master_incident_id=master_incident_id,
            similar_count=similar_count,
            embedding=embedding.tolist(),
        )

    # ── Master Incident management ────────────────────────────────────────────

    async def _get_or_create_incident(
        self, similar_count: int, sample_text: str
    ) -> str:
        """
        Return the active master incident id, creating one if none exists.
        The active incident is stored in Redis with TTL = window_secs.
        """
        active = await self.redis.get(DEDUP_ACTIVE_KEY)
        if active:
            incident_id = active.decode() if isinstance(active, bytes) else active
            # Update ticket count
            await self.redis.hincrby(DEDUP_INCIDENTS_KEY, f"{incident_id}:count", 1)
            return incident_id

        # Create new master incident
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        incident_data = json.dumps({
            "incident_id":    incident_id,
            "created_at":     time.time(),
            "similar_count":  similar_count,
            "sample_text":    sample_text[:200],
            "status":         "open",
        })
        await self.redis.hset(DEDUP_INCIDENTS_KEY, incident_id, incident_data)
        await self.redis.setex(DEDUP_ACTIVE_KEY, self.window_secs, incident_id)

        logger.info("[SemanticDedup] 📋 Master Incident created: %s", incident_id)
        return incident_id

    async def get_incident(self, incident_id: str) -> Optional[dict]:
        """Retrieve a master incident by id."""
        raw = await self.redis.hget(DEDUP_INCIDENTS_KEY, incident_id)
        if raw is None:
            return None
        return json.loads(raw)

    async def list_incidents(self) -> list:
        """Return all stored master incidents."""
        raw = await self.redis.hgetall(DEDUP_INCIDENTS_KEY)
        results = []
        for _key, value in raw.items():
            try:
                # skip internal counter keys like "INC-XXXX:count"
                data = json.loads(value)
                results.append(data)
            except Exception:
                pass
        return results
