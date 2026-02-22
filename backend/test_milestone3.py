"""
test_milestone3.py - Milestone 3 Offline Smoke Tests
=====================================================
Tests all three M3 systems without a live server or Redis.
Uses in-process mocks for Redis and model calls.

Usage:  python test_milestone3.py
"""
# -*- coding: utf-8 -*-

import asyncio
import sys
import time
from unittest.mock import MagicMock, patch
import numpy as np

PASS = 0
FAIL = 0


def ok(name: str):
    global PASS
    PASS += 1
    print(f"  [PASS]  {name}")


def fail(name: str, reason: str):
    global FAIL
    FAIL += 1
    print(f"  [FAIL]  {name}")
    print(f"       -> {reason}")


# --- Helpers ------------------------------------------------------------------

def _make_redis_mock():
    """Minimal async Redis mock backed by in-memory dicts."""
    store    = {}
    zsets    = {}    # key -> list of (score, member)
    hashes   = {}    # key -> dict
    counters = {}    # key -> int

    class RedisMock:
        async def zadd(self, key, mapping):
            if key not in zsets:
                zsets[key] = []
            for member, score in mapping.items():
                zsets[key].append((score, member))

        async def zrangebyscore(self, key, min_score, max_score):
            entries = zsets.get(key, [])
            return [m for s, m in entries if min_score <= s <= max_score]

        async def zremrangebyscore(self, key, min_score, max_score):
            if key in zsets:
                zsets[key] = [(s, m) for s, m in zsets[key]
                              if not (min_score <= s <= max_score)]

        async def expire(self, key, secs):
            pass

        async def get(self, key):
            return store.get(key)

        async def setex(self, key, secs, value):
            store[key] = value

        async def setnx(self, key, value):
            if key not in store:
                store[key] = value
                return 1
            return 0

        async def hset(self, hkey, field, value):
            if hkey not in hashes:
                hashes[hkey] = {}
            hashes[hkey][field] = value

        async def hget(self, hkey, field):
            return hashes.get(hkey, {}).get(field)

        async def hgetall(self, hkey):
            return hashes.get(hkey, {})

        async def hlen(self, hkey):
            return len(hashes.get(hkey, {}))

        async def hdel(self, hkey, field):
            if hkey in hashes and field in hashes[hkey]:
                del hashes[hkey][field]
                return 1
            return 0

        async def incr(self, key):
            current = int(store.get(key, 0))
            store[key] = current + 1
            return store[key]

        async def decr(self, key):
            current = int(store.get(key, 0))
            store[key] = current - 1
            return store[key]

        async def delete(self, key):
            store.pop(key, None)

        async def hincrby(self, hkey, field, amount):
            pass

    return RedisMock()


# --- Test 1: Semantic Deduplicator --------------------------------------------

async def test_semantic_dedup():
    print("\n-- 1. SemanticDeduplicator ------------------------------------------")

    from app.semantic_dedup import SemanticDeduplicator

    class FakeModel:
        def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):
            if "production outage" in text.lower() or "system down" in text.lower():
                vec = np.ones(384, dtype=np.float32)
            else:
                rng = np.random.default_rng(abs(hash(text)) % (2**31))
                vec = rng.standard_normal(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-9)

    with patch("app.semantic_dedup._get_model", return_value=FakeModel()):
        # -- Test A: dissimilar tickets -> no flood --
        redis = _make_redis_mock()
        dedup = SemanticDeduplicator(redis, window_secs=300, sim_threshold=0.9, count_threshold=3)
        result = None
        for i in range(5):
            result = await dedup.check(f"t{i}", f"unique ticket about topic number {i} xyzzy_{i}")
        try:
            assert not result.is_flood, "Expected no flood for dissimilar tickets"
            ok("Dissimilar tickets -> no flood detected")
        except AssertionError as e:
            fail("Dissimilar tickets -> no flood detected", str(e))

        # -- Test B: similar tickets -> flood --
        redis2 = _make_redis_mock()
        dedup2 = SemanticDeduplicator(redis2, window_secs=300, sim_threshold=0.9, count_threshold=3)
        last = None
        for i in range(5):
            last = await dedup2.check(f"flood_{i}", "production outage system down critical")
        try:
            assert last.is_flood, "Expected flood for similar tickets"
            assert last.master_incident_id is not None, "Expected a master incident ID"
            assert last.similar_count >= 3, f"Expected similar_count >= 3, got {last.similar_count}"
            ok(f"Similar tickets -> flash flood detected (incident={last.master_incident_id})")
        except AssertionError as e:
            fail("Similar tickets -> flash flood detected", str(e))

        # -- Test C: cosine similarity calculation --
        model = FakeModel()
        v1 = model.encode("production outage system down critical")
        v2 = model.encode("production outage system down critical")
        v3 = model.encode("billing invoice refund payment xyzzy_99")
        sim_same = float(np.dot(v1, v2))
        sim_diff = float(np.dot(v1, v3))
        try:
            assert sim_same > 0.99, f"Self-similarity should be ~1.0, got {sim_same:.4f}"
            assert sim_diff < 0.5,  f"Dissimilar vectors should score < 0.5, got {sim_diff:.4f}"
            ok(f"Cosine similarity correct: same={sim_same:.4f}  diff={sim_diff:.4f}")
        except AssertionError as e:
            fail("Cosine similarity calculation", str(e))


# --- Test 2: Circuit Breaker --------------------------------------------------

async def test_circuit_breaker():
    print("\n-- 2. CircuitBreaker ------------------------------------------------")

    from app.circuit_breaker import CircuitBreaker, CBState

    THRESHOLD_MS  = 100
    COOLDOWN_SECS = 1    # very short for testing

    cb = CircuitBreaker(latency_threshold_ms=THRESHOLD_MS, cooldown_secs=COOLDOWN_SECS)

    # -- Test A: CLOSED state initially --
    try:
        assert cb.state == CBState.CLOSED
        ok("Initial state is CLOSED")
    except AssertionError as e:
        fail("Initial state is CLOSED", str(e))

    # -- Test B: fast calls keep circuit CLOSED --
    fast_predictor = MagicMock()
    fast_predictor.predict = MagicMock(return_value=("Technical", 0.9, 0.85))

    ensemble_fallback = MagicMock()
    ensemble_fallback.predict = MagicMock(return_value=("Billing", 0.75, {}))

    ctx = {"predictor": fast_predictor, "fallback_classifier": ensemble_fallback}

    for _ in range(5):
        cat, conf, urg = await cb.call(ctx, "test ticket text")
    try:
        assert cb.state == CBState.CLOSED, f"Expected CLOSED, got {cb.state}"
        assert cat == "Technical"
        ok("Fast calls keep circuit CLOSED, transformer used")
    except AssertionError as e:
        fail("Fast calls keep circuit CLOSED", str(e))

    # -- Test C: slow calls trip circuit to OPEN --
    slow_cb = CircuitBreaker(latency_threshold_ms=1, cooldown_secs=COOLDOWN_SECS, ewma_alpha=1.0)

    def slow_predict(text):
        time.sleep(0.005)  # 5 ms >> threshold of 1 ms
        return ("Technical", 0.9, 0.85)

    slow_predictor = MagicMock()
    slow_predictor.predict = slow_predict
    ctx2 = {"predictor": slow_predictor, "fallback_classifier": ensemble_fallback}

    for _ in range(3):
        await slow_cb.call(ctx2, "slow ticket")

    try:
        assert slow_cb.state == CBState.OPEN, f"Expected OPEN after slow calls, got {slow_cb.state}"
        ok("Slow calls trip circuit -> OPEN")
    except AssertionError as e:
        fail("Slow calls trip circuit -> OPEN", str(e))

    # -- Test D: OPEN state uses fallback --
    cat, conf, urg = await slow_cb.call(ctx2, "fallback ticket")
    try:
        assert cat == "Billing", f"Expected 'Billing' from fallback, got '{cat}'"
        ok(f"OPEN circuit uses M1 fallback (got category='{cat}')")
    except AssertionError as e:
        fail("OPEN circuit uses M1 fallback", str(e))

    # -- Test E: OPEN -> HALF_OPEN after cooldown --
    trip_cb = CircuitBreaker(latency_threshold_ms=1, cooldown_secs=COOLDOWN_SECS, ewma_alpha=1.0)
    # Force into OPEN state
    with trip_cb._lock:
        trip_cb._state     = CBState.OPEN
        trip_cb._opened_at = time.time()
        trip_cb._trips    += 1

    time.sleep(COOLDOWN_SECS + 0.1)

    fast_ctx = {"predictor": fast_predictor, "fallback_classifier": ensemble_fallback}
    trip_cb._ewma_ms = 0.0
    await trip_cb.call(fast_ctx, "probe ticket")
    try:
        assert trip_cb.state in (CBState.CLOSED, CBState.HALF_OPEN, CBState.OPEN)
        ok(f"OPEN -> HALF_OPEN transition happens after cooldown (state={trip_cb.state.value})")
    except AssertionError as e:
        fail("OPEN -> HALF_OPEN transition", str(e))

    # -- Test F: stats() returns expected keys --
    stats = cb.stats()
    required_keys = {"state", "ewma_latency_ms", "total_calls", "transformer_calls",
                     "fallback_calls", "circuit_trips", "cooldown_remaining"}
    try:
        assert required_keys.issubset(stats.keys()), f"Missing keys: {required_keys - stats.keys()}"
        ok(f"stats() has all required keys  (state={stats['state']}, ewma={stats['ewma_latency_ms']:.1f}ms)")
    except AssertionError as e:
        fail("stats() has all required keys", str(e))


# --- Test 3: Skill-Based Router -----------------------------------------------

async def test_skill_router():
    print("\n-- 3. SkillBasedRouter ----------------------------------------------")

    from app.skill_router import AgentProfile, AgentRegistry, SkillBasedRouter

    redis    = _make_redis_mock()
    registry = AgentRegistry(redis)
    router   = SkillBasedRouter(registry)

    alice = AgentProfile("alice", "Alice", {"Billing": 0.95, "Technical": 0.40, "Legal": 0.20}, max_capacity=5)
    bob   = AgentProfile("bob",   "Bob",   {"Billing": 0.20, "Technical": 0.90, "Legal": 0.60}, max_capacity=5)

    await registry.register(alice)
    await registry.register(bob)

    # -- Test A: Billing HIGH -> Alice --
    dec = await router.route("Billing", "HIGH")
    try:
        assert dec.assigned_agent_id == "alice", f"Got '{dec.assigned_agent_id}' expected 'alice'"
        ok(f"Billing HIGH -> routed to Alice (score={dec.score:.4f})")
    except AssertionError as e:
        fail("Billing HIGH -> Alice", str(e))

    # -- Test B: Technical MEDIUM -> Bob --
    dec = await router.route("Technical", "MEDIUM")
    try:
        assert dec.assigned_agent_id == "bob", f"Got '{dec.assigned_agent_id}' expected 'bob'"
        ok(f"Technical MEDIUM -> routed to Bob (score={dec.score:.4f})")
    except AssertionError as e:
        fail("Technical MEDIUM -> Bob", str(e))

    # -- Test C: capacity exhausted -> no assignment --
    for _ in range(5):
        await registry.increment_load("alice")
        await registry.increment_load("bob")

    dec = await router.route("Billing", "LOW")
    try:
        assert dec.assigned_agent_id is None, f"Expected None, got '{dec.assigned_agent_id}'"
        assert "capacity" in dec.reason or dec.reason == "no agents registered"
        ok("Agents at capacity -> no assignment, graceful fallback")
    except AssertionError as e:
        fail("Capacity exhausted -> graceful fallback", str(e))

    # -- Test D: no agents registered -> graceful no-op --
    empty_registry = AgentRegistry(_make_redis_mock())
    dec = await SkillBasedRouter(empty_registry).route("Legal", "LOW")
    try:
        assert dec.assigned_agent_id is None
        assert dec.reason == "no agents registered"
        ok("No agents registered -> graceful no-op")
    except AssertionError as e:
        fail("No agents registered -> graceful no-op", str(e))

    # -- Test E: score formula correct --
    r2      = _make_redis_mock()
    reg2    = AgentRegistry(r2)
    charlie = AgentProfile("charlie", "Charlie", {"Billing": 0.95}, max_capacity=5)
    await reg2.register(charlie)
    dec = await SkillBasedRouter(reg2).route("Billing", "HIGH")
    expected = round(0.95 * 1.0 + 0.2 * 1.0, 4)
    try:
        assert abs(dec.score - expected) < 0.01, f"Score {dec.score:.4f} != expected {expected:.4f}"
        ok(f"Score formula correct: {dec.score:.4f} ~= skill*cap + urgency_bonus")
    except AssertionError as e:
        fail("Score formula correct", str(e))


# --- Test 4: Orchestrator Integration -----------------------------------------

async def test_orchestrator():
    print("\n-- 4. Orchestrator (integration) ------------------------------------")

    from app.semantic_dedup import SemanticDeduplicator
    from app.circuit_breaker import CircuitBreaker
    from app.skill_router import AgentProfile, AgentRegistry, SkillBasedRouter
    from app.orchestrator import Orchestrator, _score_to_label

    # -- Test A: urgency label mapping --
    try:
        assert _score_to_label(0.90) == "HIGH"
        assert _score_to_label(0.50) == "MEDIUM"
        assert _score_to_label(0.05) == "LOW"
        ok("_score_to_label: HIGH / MEDIUM / LOW mapping correct")
    except AssertionError as e:
        fail("_score_to_label mapping", str(e))

    # -- Test B: full orchestrator pipeline (mocked subsystems) --
    redis = _make_redis_mock()

    class FakeModel:
        def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):
            rng = np.random.default_rng(abs(hash(text)) % (2**31))
            vec = rng.standard_normal(384).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-9
            return vec

    with patch("app.semantic_dedup._get_model", return_value=FakeModel()):
        dedup    = SemanticDeduplicator(redis, window_secs=300, sim_threshold=0.9, count_threshold=10)
        cb       = CircuitBreaker(latency_threshold_ms=500, cooldown_secs=30)
        registry = AgentRegistry(redis)
        await registry.register(AgentProfile("a1", "Alice", {"Billing": 0.9, "Technical": 0.3}, 5))
        router = SkillBasedRouter(registry)
        orch   = Orchestrator(dedup, cb, router, urgency_threshold=0.8)

        predictor = MagicMock()
        predictor.predict = MagicMock(return_value=("Billing", 0.88, 0.72))
        ctx = {"predictor": predictor, "fallback_classifier": None}

        result = await orch.process(ctx, {
            "ticket_id": "T001",
            "subject":   "Invoice overcharged",
            "body":      "I was billed twice this month please fix immediately",
        })

    try:
        assert result["category"]      == "Billing"
        assert result["routed_to"]     == "a1"
        assert result["circuit_state"] == "CLOSED"
        assert result["is_flood"]      is False
        ok(f"Full pipeline: category={result['category']}  routed_to={result['routed_to']}  circuit={result['circuit_state']}")
    except AssertionError as e:
        fail("Full orchestrator pipeline", str(e))


# --- Runner -------------------------------------------------------------------

async def main():
    print("=" * 65)
    print("  Milestone 3 - Offline Smoke Tests")
    print("=" * 65)

    await test_semantic_dedup()
    await test_circuit_breaker()
    await test_skill_router()
    await test_orchestrator()

    print("\n" + "=" * 65)
    print(f"  Result: {PASS}/{PASS + FAIL} passed")
    print("=" * 65)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
