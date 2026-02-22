"""
SkillBasedRouter — Milestone 3
================================
Stateful agent registry with constraint-optimization routing.

Every agent has a "skill vector":
    { "Billing": 0.9, "Technical": 0.5, "Legal": 0.3 }

Routing Algorithm
-----------------
For a ticket with (category, urgency):

    capacity_weight(agent) = max(0, (max_capacity - active_tickets) / max_capacity)
    skill_score(agent)     = agent.skills.get(category, 0.0)
    urgency_bonus          = { "HIGH": 0.2, "MEDIUM": 0.1, "LOW": 0.0 }[urgency]

    total_score(agent) = skill_score × capacity_weight + urgency_bonus × capacity_weight

Pick the agent with the highest total_score among agents that have
    active_tickets < max_capacity.

Ties are broken by agent_id (alphabetical) for determinism.

Persistence
-----------
Each agent profile is stored in Redis as a hash field:
    agents:{agent_id}  → JSON blob

Active ticket counts are tracked in Redis too:
    agent_load:{agent_id}  → integer (INCR / DECR)

This lets multiple worker processes share state.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

AGENTS_HASH_KEY  = "agents:registry"
AGENT_LOAD_PREFIX = "agent:load:"

# Urgency bonus weights
URGENCY_BONUS: Dict[str, float] = {
    "HIGH":   0.2,
    "MEDIUM": 0.1,
    "LOW":    0.0,
}


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class AgentProfile:
    agent_id:     str
    name:         str
    skills:       Dict[str, float]   # category → proficiency [0.0, 1.0]
    max_capacity: int = 10
    active_tickets: int = 0          # live count from Redis at query time

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentProfile":
        return cls(
            agent_id=d["agent_id"],
            name=d["name"],
            skills=d["skills"],
            max_capacity=d.get("max_capacity", 10),
            active_tickets=d.get("active_tickets", 0),
        )


# ─── Routing Decision ─────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    assigned_agent_id:   Optional[str]
    assigned_agent_name: Optional[str]
    score:               float
    reason:              str           # e.g. "best skill match" or "no agents available"
    all_scores:          Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── AgentRegistry ────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Redis-backed registry of support agents.
    Provides CRUD + real-time load tracking.
    """

    def __init__(self, redis_client):
        self.redis = redis_client

    # ── Registration ──────────────────────────────────────────────────────────

    async def register(self, profile: AgentProfile) -> None:
        """Add or update an agent in the registry."""
        data = {k: v for k, v in profile.to_dict().items() if k != "active_tickets"}
        await self.redis.hset(AGENTS_HASH_KEY, profile.agent_id, json.dumps(data))
        # Initialize load counter only if it doesn't exist
        await self.redis.setnx(f"{AGENT_LOAD_PREFIX}{profile.agent_id}", 0)
        logger.info("[AgentRegistry] Registered agent %s (%s)", profile.agent_id, profile.name)

    async def unregister(self, agent_id: str) -> bool:
        """Remove an agent. Returns True if it existed."""
        removed = await self.redis.hdel(AGENTS_HASH_KEY, agent_id)
        await self.redis.delete(f"{AGENT_LOAD_PREFIX}{agent_id}")
        if removed:
            logger.info("[AgentRegistry] Unregistered agent %s", agent_id)
        return bool(removed)

    # ── Load tracking ─────────────────────────────────────────────────────────

    async def increment_load(self, agent_id: str) -> int:
        """Mark one more active ticket assigned to agent. Returns new count."""
        return int(await self.redis.incr(f"{AGENT_LOAD_PREFIX}{agent_id}"))

    async def decrement_load(self, agent_id: str) -> int:
        """Mark one ticket resolved for agent. Returns new count (min 0)."""
        count = int(await self.redis.decr(f"{AGENT_LOAD_PREFIX}{agent_id}"))
        if count < 0:
            await self.redis.set(f"{AGENT_LOAD_PREFIX}{agent_id}", 0)
            return 0
        return count

    # ── Queries ───────────────────────────────────────────────────────────────

    async def get_all(self) -> List[AgentProfile]:
        """Return all registered agents with live load counts."""
        raw_all = await self.redis.hgetall(AGENTS_HASH_KEY)
        profiles: List[AgentProfile] = []
        for agent_id_bytes, data_bytes in raw_all.items():
            try:
                agent_id = agent_id_bytes.decode() if isinstance(agent_id_bytes, bytes) else agent_id_bytes
                data     = json.loads(data_bytes)
                load_raw = await self.redis.get(f"{AGENT_LOAD_PREFIX}{agent_id}")
                active   = int(load_raw) if load_raw is not None else 0
                data["active_tickets"] = active
                profiles.append(AgentProfile.from_dict(data))
            except Exception as exc:
                logger.warning("[AgentRegistry] Failed to parse agent record: %s", exc)
        return profiles

    async def get(self, agent_id: str) -> Optional[AgentProfile]:
        """Return a single agent profile (with live load) or None."""
        raw = await self.redis.hget(AGENTS_HASH_KEY, agent_id)
        if raw is None:
            return None
        data = json.loads(raw)
        load_raw = await self.redis.get(f"{AGENT_LOAD_PREFIX}{agent_id}")
        data["active_tickets"] = int(load_raw) if load_raw is not None else 0
        return AgentProfile.from_dict(data)

    async def count(self) -> int:
        return await self.redis.hlen(AGENTS_HASH_KEY)


# ─── SkillBasedRouter ─────────────────────────────────────────────────────────

class SkillBasedRouter:
    """
    Constraint-optimisation router.
    Instantiate once; call .route() per ticket.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    async def route(
        self,
        category: str,
        urgency:  str,
    ) -> RoutingDecision:
        """
        Find the best available agent for the given (category, urgency) pair.

        Returns a RoutingDecision.  If no agents are registered or all are at
        capacity, assigned_agent_id is None.
        """
        agents = await self.registry.get_all()

        if not agents:
            return RoutingDecision(
                assigned_agent_id=None,
                assigned_agent_name=None,
                score=0.0,
                reason="no agents registered",
            )

        bonus = URGENCY_BONUS.get(urgency.upper(), 0.0)

        best_agent: Optional[AgentProfile] = None
        best_score: float                  = -1.0
        all_scores: Dict[str, float]       = {}

        for agent in sorted(agents, key=lambda a: a.agent_id):    # deterministic tie-breaking
            # Capacity constraint
            if agent.active_tickets >= agent.max_capacity:
                all_scores[agent.agent_id] = -1.0
                continue

            cap_weight   = (agent.max_capacity - agent.active_tickets) / agent.max_capacity
            skill_score  = agent.skills.get(category, 0.0)
            total        = skill_score * cap_weight + bonus * cap_weight
            all_scores[agent.agent_id] = round(total, 4)

            if total > best_score:
                best_score = total
                best_agent = agent

        if best_agent is None:
            return RoutingDecision(
                assigned_agent_id=None,
                assigned_agent_name=None,
                score=0.0,
                reason="all agents at capacity",
                all_scores=all_scores,
            )

        # Increment load counter
        new_load = await self.registry.increment_load(best_agent.agent_id)
        logger.info(
            "[Router] Ticket routed → agent=%s  category=%s  urgency=%s  "
            "score=%.4f  load=%d/%d",
            best_agent.agent_id, category, urgency,
            best_score, new_load, best_agent.max_capacity,
        )

        return RoutingDecision(
            assigned_agent_id=best_agent.agent_id,
            assigned_agent_name=best_agent.name,
            score=round(best_score, 4),
            reason="best skill match",
            all_scores=all_scores,
        )

    async def release_ticket(self, agent_id: str) -> None:
        """Call when a ticket is resolved to decrement agent load."""
        await self.registry.decrement_load(agent_id)
        logger.info("[Router] Released ticket from agent %s", agent_id)
