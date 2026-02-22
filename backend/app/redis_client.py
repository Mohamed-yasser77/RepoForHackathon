"""
Redis Client + Redlock — Milestone 2
=====================================
Async Redis connection pool and a Redlock implementation for
distributed deduplication of identical ticket_id submissions.

The Redlock algorithm ensures that when 10 identical ticket_ids
arrive in the same millisecond, only ONE is pushed to the ARQ
worker — but all 10 HTTP requests safely return 202 Accepted.

Implementation follows the single-instance Redlock pattern using:
  SET resource_name unique_value NX PX ttl_ms
  DEL via Lua script (atomic compare-and-delete)
"""

import logging
import uuid
from typing import Optional

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Lua script for atomic lock release ───────────────────────────────────────
# Only deletes the key if the stored value matches our unique lock token.
# This prevents a client from accidentally releasing another client's lock.

_RELEASE_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""


class RedlockManager:
    """
    Single-instance Redlock for distributed mutual exclusion.

    Usage:
        rl = RedlockManager(redis_pool)
        token = await rl.acquire("ticket_lock:tkt_abc123", ttl_ms=5000)
        if token:
            # ... enqueue the job — we hold the lock ...
            await rl.release("ticket_lock:tkt_abc123", token)
        else:
            # Another process already enqueued this ticket
            pass
    """

    def __init__(self, redis: aioredis.Redis):
        self._redis = redis
        self._release_script: Optional[object] = None

    async def _get_release_script(self):
        """Lazily register the Lua release script."""
        if self._release_script is None:
            self._release_script = self._redis.register_script(_RELEASE_LUA)
        return self._release_script

    async def acquire(
        self,
        resource: str,
        ttl_ms: int = settings.REDLOCK_TTL_MS,
    ) -> Optional[str]:
        """
        Attempt to acquire a distributed lock on `resource`.

        Args:
            resource: The Redis key to lock (e.g., "ticket_lock:tkt_abc").
            ttl_ms:   Lock time-to-live in milliseconds.

        Returns:
            A unique lock token (str) if acquired, or None if the lock
            is already held by another process.
        """
        token = str(uuid.uuid4())
        acquired = await self._redis.set(
            resource,
            token,
            nx=True,   # SET if Not eXists
            px=ttl_ms, # auto-expire in milliseconds
        )
        if acquired:
            logger.debug("Redlock ACQUIRED: %s (token=%s, ttl=%dms)", resource, token, ttl_ms)
            return token
        else:
            logger.debug("Redlock DENIED: %s — already held", resource)
            return None

    async def release(self, resource: str, token: str) -> bool:
        """
        Release a lock only if we still own it (compare-and-delete).

        Args:
            resource: The Redis key to unlock.
            token:    The token returned by acquire().

        Returns:
            True if the lock was successfully released, False otherwise.
        """
        script = await self._get_release_script()
        result = await script(keys=[resource], args=[token])
        released = bool(result)
        if released:
            logger.debug("Redlock RELEASED: %s", resource)
        else:
            logger.warning("Redlock release FAILED (expired or stolen): %s", resource)
        return released


# ─── Connection Factory ──────────────────────────────────────────────────────

async def create_redis_pool() -> aioredis.Redis:
    """
    Create and return an async Redis connection pool.
    Supports both local redis:// and cloud rediss:// (TLS) URLs (e.g. Upstash).
    """
    url = settings.REDIS_URL
    use_ssl = url.startswith("rediss://")

    pool = aioredis.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
        **({"ssl_cert_reqs": None} if use_ssl else {}),
    )
    # Verify connectivity
    await pool.ping()
    logger.info("Redis connection established: %s", url.split("@")[-1] if "@" in url else url)
    return pool
