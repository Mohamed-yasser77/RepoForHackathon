"""
Smart-Support Ticket Routing Engine — MVR
==========================================
  • EnsembleIRClassifier  — TF-IDF + BIM + BM25, soft-vote
  • Regex urgency heuristic → HIGH / MEDIUM / LOW + score ∈ [0, 1]
  • PriorityQueueSingleton — thread-safe heapq singleton
  • In-memory Agents store  — CRUD + skill-based routing
  • In-memory Incidents store — flash-flood master incidents
  • FastAPI + Pydantic     — strict payload validation
  • CORS enabled for Next.js frontend (localhost:3000)
"""

import re
import time
import uuid
import hashlib
from contextlib import asynccontextmanager
from typing import Optional, Dict, List

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classifier import EnsembleIRClassifier
from queue_manager import PriorityQueueSingleton


# ─── In-memory stores ─────────────────────────────────────────────────────────

_agents: Dict[str, dict] = {}
_incidents: Dict[str, dict] = {}
_submitted_ticket_ids: set = set()   # for duplicate detection


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
        "Ensemble IR Ticket Router\n\n"
        "Classifier: TF-IDF·LogReg (0.45) + BIM·BernoulliNB (0.25) + BM25 (0.30)\n"
        "Dataset:    Waseem Alastal — Customer Support Ticket Dataset (Kaggle)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js frontend
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Add production origins from environment variable if present
env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
if env_origins:
    # Expecting a comma-separated list: "https://my-app.vercel.app,https://api.my-app.com"
    allowed_origins.extend([o.strip() for o in env_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

# ── Health ──

class HealthResponse(BaseModel):
    status: str
    queue_size: int
    classifier: str
    redis: str = "disconnected"          # stub — no Redis in MVR
    app_version: str = "1.0.0"
    circuit_breaker_state: str = "CLOSED"
    agents_online: int = 0


# ── Tickets ──

class TicketRequest(BaseModel):
    ticket_id: Optional[str] = Field(default=None, example="TKT-001")
    subject:   str           = Field(..., min_length=1, max_length=300,
                                     example="Invoice overcharged me twice")
    body:      str           = Field(..., min_length=1, max_length=5000,
                                     example="I was billed twice this month. Fix ASAP.")
    customer_id: Optional[str] = Field(default=None, example="cust_1234")
    channel:     Optional[str] = Field(
        default="email",
        example="email",
        pattern="^(email|chat|phone|web)$"
    )


class TicketResponse(BaseModel):
    ticket_id:        str
    status:           str           # "accepted" | "duplicate"
    message:          str
    duplicate:        bool
    category:         str
    urgency:          str           # HIGH | MEDIUM | LOW
    urgency_score:    float
    confidence:       float
    queue_position:   int
    classifier_votes: dict
    timestamp:        float
    # Routing fields
    routed_to:        Optional[str] = None
    routing_score:    float = 0.0
    routing_reason:   Optional[str] = None


# ── Agents ──

class AgentSkills(BaseModel):
    Billing:   float = Field(0.5, ge=0.0, le=1.0)
    Technical: float = Field(0.5, ge=0.0, le=1.0)
    Legal:     float = Field(0.5, ge=0.0, le=1.0)

class AgentPayload(BaseModel):
    agent_id:     str
    name:         str
    skills:       AgentSkills
    max_capacity: int = Field(5, ge=1, le=50)

class Agent(BaseModel):
    agent_id:       str
    name:           str
    skills:         AgentSkills
    max_capacity:   int
    active_tickets: int = 0
    assigned_tickets: List[dict] = []

class AgentsResponse(BaseModel):
    agents: List[Agent]
    total:  int

class ReleaseResponse(BaseModel):
    agent_id:       str
    active_tickets: int
    assigned_tickets: List[dict] = []


# ── Incidents ──

class Incident(BaseModel):
    incident_id:  str
    created_at:   float   # UNIX timestamp
    similar_count: int
    sample_text:  str
    status:       str     # "open" | "closed"

class IncidentsResponse(BaseModel):
    incidents: List[Incident]
    total:     int

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

# ─── Skill-Based Router (CSP) Logic ───────────────────────────────────────────

def find_best_agent(category: str, urgency: str) -> dict:
    """
    Constraint-Optimization Router (CSP).
    Finds the best available agent based on skill match and remaining capacity.
    """
    bonus = {"HIGH": 0.2, "MEDIUM": 0.1, "LOW": 0.0}.get(urgency, 0.0)
    
    best_agent_id = None
    best_score = -1.0
    reason = "no agents available"
    
    # Sort by agent_id for deterministic tie-breaking
    sorted_agent_ids = sorted(_agents.keys())
    
    for aid in sorted_agent_ids:
        agent = _agents[aid]
        
        # 1. Capacity Constraint
        if agent["active_tickets"] >= agent["max_capacity"]:
            continue
            
        # 2. Scoring (Skill + Capacity + Urgency)
        cap_weight = (agent["max_capacity"] - agent["active_tickets"]) / agent["max_capacity"]
        skill_score = agent["skills"].get(category, 0.5) # Default to 0.5 if category unknown
        
        # CSP Objective Function:
        score = (skill_score * cap_weight) + (bonus * cap_weight)
        
        if score > best_score:
            best_score = score
            best_agent_id = aid
            reason = "best skill/capacity match"

    if best_agent_id:
        return {
            "agent_id": best_agent_id,
            "score": round(best_score, 4),
            "reason": reason
        }
    
    return {
        "agent_id": None,
        "score": 0.0,
        "reason": "all agents at capacity" if _agents else "no agents registered"
    }


# ─── Routes — System ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    q: PriorityQueueSingleton = app.state.queue
    return HealthResponse(
        status="ok",
        queue_size=q.size(),
        classifier="EnsembleIR — TF-IDF·LogReg | BIM·BernoulliNB | BM25",
        agents_online=len(_agents),
    )


@app.get("/queue/stats", tags=["System"])
def queue_stats():
    """Live queue statistics by urgency and category."""
    return app.state.queue.stats()


# ─── Routes — Tickets ─────────────────────────────────────────────────────────

@app.post("/tickets", response_model=TicketResponse, status_code=201, tags=["Tickets"])
def submit_ticket(req: TicketRequest):
    """
    Submit a support ticket.
    - Detects duplicates by ticket_id within the current session.
    - Classifies into Billing / Technical / Legal using the IR ensemble.
    - Assigns urgency via regex heuristic.
    - Pushes onto the priority heap (HIGH first).
    - Routes to the best available agent via CSP scoring.
    """
    full_text = f"{req.subject} {req.body}"

    # 1. Duplicate check
    effective_id = req.ticket_id or str(uuid.uuid4())
    is_duplicate = effective_id in _submitted_ticket_ids

    if not is_duplicate:
        _submitted_ticket_ids.add(effective_id)

    # 2. Classify
    clf: EnsembleIRClassifier = app.state.classifier
    category, confidence, votes = clf.predict(full_text)

    # 3. Urgency
    urgency_label, urgency_score = compute_urgency(full_text)

    # 4. Routing (CSP)
    routing = find_best_agent(category, urgency_label)
    if routing["agent_id"]:
        agent = _agents[routing["agent_id"]]
        agent["active_tickets"] += 1
        # Store metadata for display
        agent["assigned_tickets"].append({
            "ticket_id": effective_id,
            "subject": req.subject,
            "category": category,
            "urgency": urgency_label,
            "timestamp": time.time()
        })

    # 5. Heap priority: HIGH=1, MEDIUM=2, LOW=3
    priority_map = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
    priority     = priority_map[urgency_label]

    # 6. Enqueue
    ticket_id = effective_id
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
        "duplicate":     is_duplicate,
        "routed_to":     routing["agent_id"],
    }

    queue: PriorityQueueSingleton = app.state.queue
    pos = queue.push(priority, timestamp, payload)

    # 7. Auto-create incident if we detect a flash-flood pattern
    _maybe_create_incident(category, urgency_label, full_text)

    return TicketResponse(
        ticket_id        = ticket_id,
        status           = "duplicate" if is_duplicate else "accepted",
        message          = (
            "Ticket already submitted in this session — marked as duplicate."
            if is_duplicate
            else f"Ticket enqueued. Routed to {routing['agent_id'] or 'None'}."
        ),
        duplicate        = is_duplicate,
        category         = category,
        urgency          = urgency_label,
        urgency_score    = urgency_score,
        confidence       = round(confidence, 4),
        queue_position   = pos,
        classifier_votes = votes,
        timestamp        = timestamp,
        routed_to        = routing["agent_id"],
        routing_score    = routing["score"],
        routing_reason   = routing["reason"],
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


# ─── Routes — Agents ──────────────────────────────────────────────────────────

@app.get("/agents", response_model=AgentsResponse, tags=["Agents"])
def list_agents():
    """Return all registered agents."""
    agents = [Agent(**a) for a in _agents.values()]
    return AgentsResponse(agents=agents, total=len(agents))


@app.post("/agents", response_model=Agent, status_code=201, tags=["Agents"])
def register_agent(payload: AgentPayload):
    """Register a new agent (or update if agent_id already exists)."""
    existing = _agents.get(payload.agent_id, {})
    agent_data = {
        "agent_id":       payload.agent_id,
        "name":           payload.name,
        "skills":         payload.skills.model_dump(),
        "max_capacity":   payload.max_capacity,
        "active_tickets": existing.get("active_tickets", 0),
        "assigned_tickets": existing.get("assigned_tickets", []),
    }
    _agents[payload.agent_id] = agent_data
    return Agent(**agent_data)


@app.get("/agents/{agent_id}", response_model=Agent, tags=["Agents"])
def get_agent(agent_id: str):
    """Get a single agent by ID."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return Agent(**_agents[agent_id])


@app.post("/agents/{agent_id}/release", response_model=ReleaseResponse, tags=["Agents"])
def release_agent(agent_id: str):
    """Decrement active_tickets by 1 (simulates completing a ticket assignment)."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    agent = _agents[agent_id]
    agent["active_tickets"] = max(0, agent["active_tickets"] - 1)
    
    # Remove the oldest ticket if list isn't empty
    if agent["assigned_tickets"]:
        agent["assigned_tickets"].pop(0)
        
    return ReleaseResponse(
        agent_id=agent_id, 
        active_tickets=agent["active_tickets"],
        assigned_tickets=agent["assigned_tickets"]
    )


@app.delete("/agents/{agent_id}", status_code=204, tags=["Agents"])
def delete_agent(agent_id: str):
    """Remove an agent from the roster."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    del _agents[agent_id]


# ─── Routes — Incidents ───────────────────────────────────────────────────────

@app.get("/incidents", response_model=IncidentsResponse, tags=["Incidents"])
def list_incidents():
    """Return all detected master incidents (flash-flood groups)."""
    incidents = [Incident(**i) for i in _incidents.values()]
    return IncidentsResponse(incidents=incidents, total=len(incidents))


@app.get("/incidents/{incident_id}", response_model=Incident, tags=["Incidents"])
def get_incident(incident_id: str):
    """Get a single incident by ID."""
    if incident_id not in _incidents:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found")
    return Incident(**_incidents[incident_id])


# ─── Internal helpers ─────────────────────────────────────────────────────────

# Track ticket counts per (category, urgency) window for flash-flood detection
_window_counts: Dict[str, List[float]] = {}

def _maybe_create_incident(category: str, urgency: str, sample_text: str):
    """
    Create or update a master incident when ≥10 HIGH-urgency tickets of the same
    category arrive within 5 minutes (simplified flash-flood detector for MVR).
    """
    if urgency != "HIGH":
        return

    key = f"{category}:{urgency}"
    now = time.time()
    window = 300  # 5 minutes

    timestamps = _window_counts.get(key, [])
    timestamps = [t for t in timestamps if now - t < window]
    timestamps.append(now)
    _window_counts[key] = timestamps

    count = len(timestamps)
    if count >= 10:
        # Stable hash ID so the same flood maps to the same incident
        inc_id = "INC-" + hashlib.md5(key.encode()).hexdigest()[:8].upper()
        if inc_id in _incidents:
            _incidents[inc_id]["similar_count"] = count
        else:
            _incidents[inc_id] = {
                "incident_id":   inc_id,
                "created_at":    now,
                "similar_count": count,
                "sample_text":   sample_text[:200],
                "status":        "open",
            }
