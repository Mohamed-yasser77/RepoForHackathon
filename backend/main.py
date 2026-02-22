"""
Smart-Support Ticket Routing Engine — Milestone 1 (MVR)
========================================================
  • EnsembleIRClassifier  — TF-IDF + BIM + BM25, soft-vote
  • Regex urgency heuristic → HIGH / MEDIUM / LOW + score ∈ [0, 1]
  • PriorityQueueSingleton — thread-safe heapq singleton
  • FastAPI + Pydantic     — strict payload validation
"""

import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from classifier import EnsembleIRClassifier
from queue_manager import PriorityQueueSingleton


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initialising ensemble classifier …")
    clf = EnsembleIRClassifier()
    clf.load_or_train()                         # kagglehub download → train → cache
    app.state.classifier = clf
    app.state.queue      = PriorityQueueSingleton.get_instance()
    print("[Startup] ✓ Server ready.\n")
    yield
    print("[Shutdown] Bye.")


app = FastAPI(
    title="Smart-Support MVR",
    description=(
        "Milestone 1 — Ensemble IR Ticket Router\n\n"
        "Classifier: TF-IDF·LogReg (0.45) + BIM·BernoulliNB (0.25) + BM25 (0.30)\n"
        "Dataset:    Waseem Alastal — Customer Support Ticket Dataset (Kaggle)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class TicketRequest(BaseModel):
    subject:     str           = Field(..., min_length=1, max_length=300,
                                       example="Invoice overcharged me twice")
    body:        str           = Field(..., min_length=1, max_length=5000,
                                       example="I was billed twice this month. Fix ASAP.")
    customer_id: Optional[str] = Field(default=None, example="cust_1234")
    channel:     Optional[str] = Field(default="email",
                                       example="email",
                                       pattern="^(email|chat|phone|web)$")

class TicketResponse(BaseModel):
    ticket_id:        str
    category:         str
    urgency:          str           # HIGH | MEDIUM | LOW
    urgency_score:    float         # 0.0 – 1.0
    confidence:       float         # ensemble blended confidence
    queue_position:   int
    classifier_votes: dict          # per-model breakdown
    timestamp:        float


# ─── Urgency Heuristic ────────────────────────────────────────────────────────

_HIGH = re.compile(
    r"\b(asap|urgent|urgently|immediately|critical|emergency|broken|outage|"
    r"not working|cannot access|data loss|security breach|compromised|"
    r"lawsuit|legal action|refund now|escalate|production|sev[- ]?1|"
    r"threatening|down|unresponsive|unreachable)\b",
    re.IGNORECASE,
)
_MED = re.compile(
    r"\b(slow|delay|error|fail|issue|problem|bug|incorrect|wrong|"
    r"disappointed|frustrated|annoyed|waiting|pending|days|week)\b",
    re.IGNORECASE,
)

def compute_urgency(text: str) -> tuple[str, float]:
    """Return (label, score) where score ∈ [0, 1]."""
    h = len(_HIGH.findall(text))
    m = len(_MED.findall(text))

    if h >= 2:
        return "HIGH",   round(min(0.95, 0.75 + 0.05 * h),  3)
    if h == 1:
        return "HIGH",   round(min(0.74, 0.55 + 0.04 * m),  3)
    if m >= 2:
        return "MEDIUM", round(min(0.54, 0.35 + 0.04 * m),  3)
    if m == 1:
        return "MEDIUM", 0.30
    return "LOW", 0.10


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    q: PriorityQueueSingleton = app.state.queue
    return {
        "status":     "ok",
        "queue_size": q.size(),
        "classifier": "EnsembleIR — TF-IDF·LogReg | BIM·BernoulliNB | BM25",
    }


@app.post("/tickets", response_model=TicketResponse, status_code=201, tags=["Tickets"])
def submit_ticket(req: TicketRequest):
    """
    Submit a support ticket.
    - Classifies into Billing / Technical / Legal using the IR ensemble.
    - Assigns urgency via regex heuristic.
    - Pushes onto the priority heap (HIGH first).
    - Returns ticket ID, classification, urgency, and queue position.
    """
    full_text = f"{req.subject} {req.body}"

    # 1. Classify
    clf: EnsembleIRClassifier = app.state.classifier
    category, confidence, votes = clf.predict(full_text)

    # 2. Urgency
    urgency_label, urgency_score = compute_urgency(full_text)

    # 3. Heap priority: HIGH=1, MEDIUM=2, LOW=3
    priority_map = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
    priority     = priority_map[urgency_label]

    # 4. Enqueue
    ticket_id = str(uuid.uuid4())
    timestamp = time.time()
    payload   = {
        "ticket_id":     ticket_id,
        "category":      category,
        "urgency":       urgency_label,
        "urgency_score": urgency_score,
        "confidence":    round(confidence, 4),
        "subject":       req.subject,
        "body":          req.body,
        "customer_id":   req.customer_id,
        "channel":       req.channel,
        "timestamp":     timestamp,
    }

    queue: PriorityQueueSingleton = app.state.queue
    pos = queue.push(priority, timestamp, payload)

    return TicketResponse(
        ticket_id        = ticket_id,
        category         = category,
        urgency          = urgency_label,
        urgency_score    = urgency_score,
        confidence       = round(confidence, 4),
        queue_position   = pos,
        classifier_votes = votes,
        timestamp        = timestamp,
    )


@app.get("/tickets/peek", tags=["Tickets"])
def peek_ticket():
    """Peek at the next ticket without removing it."""
    q = app.state.queue
    item = q.peek()
    if item is None:
        raise HTTPException(status_code=404, detail="Queue is empty")
    return item


@app.post("/tickets/dequeue", tags=["Tickets"])
def dequeue_ticket():
    """Pop the highest-priority ticket (simulates agent pickup)."""
    q = app.state.queue
    item = q.pop()
    if item is None:
        raise HTTPException(status_code=404, detail="Queue is empty")
    return item


@app.get("/queue/stats", tags=["System"])
def queue_stats():
    """Live queue statistics by urgency and category."""
    return app.state.queue.stats()