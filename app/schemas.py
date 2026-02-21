"""
Pydantic Schemas — Milestone 2
==============================
Strict payload validation for the async ticket routing API.
"""

from typing import Optional

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


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    redis: str
    app_version: str
