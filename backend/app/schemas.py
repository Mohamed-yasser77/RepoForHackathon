"""
Pydantic Schemas — Milestone 3
==============================
Strict payload validation for the async ticket routing API.
Includes M3 schemas: AgentProfile, RoutingDecision, MasterIncident.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Request Payloads ─────────────────────────────────────────────────────────

class TicketPayload(BaseModel):
    """Incoming ticket submission payload."""

    ticket_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique ticket identifier (client-generated; used for dedup)",
        examples=["tkt_a1b2c3d4"],
    )
    subject: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Short ticket subject line",
        examples=["Invoice overcharged me twice"],
    )
    body: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Full ticket description",
        examples=["I was billed twice this month. Fix ASAP."],
    )
    customer_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Customer identifier",
        examples=["cust_1234"],
    )
    channel: Optional[str] = Field(
        default="email",
        pattern=r"^(email|chat|phone|web)$",
        description="Submission channel",
        examples=["email"],
    )


# ─── Response Payloads ────────────────────────────────────────────────────────

class AcceptedResponse(BaseModel):
    """HTTP 202 Accepted response confirming ticket receipt."""

    ticket_id: str = Field(
        ..., description="Echo of the submitted ticket ID"
    )
    status: str = Field(
        default="accepted",
        description="Processing status — always 'accepted' for 202",
    )
    message: str = Field(
        default="Ticket enqueued for background processing.",
        description="Human-readable status message",
    )
    duplicate: bool = Field(
        default=False,
        description="True if this ticket_id was already enqueued (dedup by Redlock)",
    )
    # ── Milestone 3 fields (filled in async by the worker) ────────────────
    routed_to:   Optional[str] = Field(
        default=None,
        description="Agent ID the ticket will be routed to (async, may be null)",
    )
    incident_id: Optional[str] = Field(
        default=None,
        description="Master Incident ID if a flash flood was detected",
    )


class HealthResponse(BaseModel):
    """Health-check response — extended for Milestone 3."""

    status:               str
    redis:                str
    app_version:          str
    circuit_breaker_state: Optional[str]  = None    # CLOSED | OPEN | HALF_OPEN
    circuit_ewma_ms:       Optional[float] = None
    agents_online:         Optional[int]   = None


# ─── Milestone 3 Schemas ──────────────────────────────────────────────────────

class AgentProfileRequest(BaseModel):
    """Payload for registering a new support agent."""

    agent_id: str = Field(
        ..., min_length=1, max_length=64,
        description="Unique agent identifier",
        examples=["agent_007"],
    )
    name: str = Field(
        ..., min_length=1, max_length=128,
        description="Human-readable agent name",
        examples=["Alice"],
    )
    skills: Dict[str, float] = Field(
        ...,
        description="Skill vector: category → proficiency [0.0–1.0]",
        examples=[{"Billing": 0.9, "Technical": 0.5, "Legal": 0.3}],
    )
    max_capacity: int = Field(
        default=10, ge=1, le=100,
        description="Maximum concurrent tickets this agent can handle",
    )


class AgentProfileResponse(BaseModel):
    """Full agent profile with live load stats."""

    agent_id:       str
    name:           str
    skills:         Dict[str, float]
    max_capacity:   int
    active_tickets: int


class AgentListResponse(BaseModel):
    """List of all registered agents."""

    agents: List[AgentProfileResponse]
    total:  int


class RoutingDecisionResponse(BaseModel):
    """Result of the skill-based routing algorithm."""

    assigned_agent_id:   Optional[str]
    assigned_agent_name: Optional[str]
    score:               float
    reason:              str
    all_scores:          Dict[str, float] = {}


class MasterIncidentResponse(BaseModel):
    """A master incident created during a flash-flood event."""

    incident_id:   str
    created_at:    float
    similar_count: int
    sample_text:   str
    status:        str   # open | resolved
