# Smart-Support API Documentation (Full Project)

This document outlines the complete REST API for the Smart-Support Autonomous Orchestrator. The system is built with FastAPI and integrates features from Milestone 1 (M1), Milestone 2 (M2), and Milestone 3 (M3).

## Base Information
- **Base URL:** `http://localhost:8000`
- **Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)
- **Data Format:** Content-Type `application/json`

---

## 1. System Health & Status

### `GET /health`
Retrieves the overall health of the entire system, including the M3 components (Circuit Breaker and Agent Registry).

**Response (200 OK):**
```json
{
  "status": "ok",
  "redis": "connected",
  "app_version": "3.0.0",
  "circuit_breaker_state": "CLOSED",
  "agents_online": 3
}
```

---

## 2. Core Ticket Submission (M2 & M3)

### `POST /tickets`
This is the primary endpoint for the frontend to submit new customer support tickets. 
- Returns immediately (`202 Accepted`).
- Enqueues the ticket for asynchronous processing via ARQ.
- Triggers the full M3 pipeline (Semantic Deduplication -> Circuit Breaker -> Transformer Model -> Skill-Based Routing).
- Implements **Redlock** to instantly detect exact duplicates submitted within a 5-second window.

**Request Body:**
```json
{
  "ticket_id": "TKT-001",                   // Required: Unique ID from your frontend/DB
  "subject": "System crash",                // Required
  "body": "The entire database cluster went down and users can't login.", // Required
  "customer_id": "CUST-123",              // Optional
  "channel": "email"                        // Optional
}
```

**Response (202 Accepted):**
```json
{
  "ticket_id": "TKT-001",
  "status": "accepted",
  "message": "Ticket enqueued for M3 orchestrator pipeline.",
  "duplicate": false    // Will be true if ticket_id is locked by Redlock
}
```

---

## 3. Flash-Flood Detection (M3)
*When >10 highly similar tickets (Cosine Similarity > 0.9 via Sentence Transformers) are received within 5 minutes, they are grouped into a Master Incident to prevent alert spam.*

### `GET /incidents`
Lists all active flash-flood "Master Incidents" detected by the Semantic Deduplicator.

**Response (200 OK):**
```json
{
  "incidents": [
    {
      "incident_id": "INC-56F18251",      // Auto-generated hash ID
      "created_at": 1771743421.84,        // UNIX Timestamp
      "similar_count": 12,                // Number of tickets grouped here
      "sample_text": "System crash. The entire database cluster went...",
      "status": "open"
    }
  ],
  "total": 1
}
```

### `GET /incidents/{incident_id}`
Retrieves details for a specific Master Incident by its ID.

---

## 4. Agent Management & Skill-Based Routing (M3)
*To map tickets to humans, agents must be registered with their specific "Skill Vectors" (0.0 to 1.0). The system uses Constraint Optimization to route to the best available agent.*

### `POST /agents`
Registers a new support agent in the routing pool. **The frontend should call this when a support agent logs into the system.**

**Request Body:**
```json
{
  "agent_id": "alice_super_tech",  // Unique string identifying the agent
  "name": "Alice",                 // Display name
  "skills": {
    "Billing": 0.1,
    "Technical": 0.95,           // Array of categories matching the ML model output
    "Legal": 0.4
  },
  "max_capacity": 5                // Maximum concurrent tickets they can handle
}
```

**Response (201 Created):**
*(Returns the registered agent object, including initialized `active_tickets: 0`)*


### `GET /agents`
Lists all registered agents and their real-time ticket load. Used to populate an "Online Agents" or Admin Dashboard on the frontend.

**Response (200 OK):**
```json
{
  "agents": [
    {
      "agent_id": "alice_super_tech",
      "name": "Alice",
      "skills": {"Billing": 0.1, "Technical": 0.95, "Legal": 0.4},
      "max_capacity": 5,
      "active_tickets": 2          // Currently working on 2 tickets
    }
  ],
  "total": 1
}
```

### `GET /agents/{agent_id}`
Retrieves the profile and active load for a specific single agent.


### `POST /agents/{agent_id}/release`
Decrements the `active_tickets` count for an agent. **CRITICAL: The frontend MUST call this endpoint whenever an agent marks a ticket as "Resolved" or "Closed" in the UI.** Failing to do this means the agent remains at max capacity permanently.

**Response (200 OK):**
```json
{
  "agent_id": "alice_super_tech",
  "active_tickets": 1              // Updated load count
}
```

### `DELETE /agents/{agent_id}`
Unregisters an agent from the system so they stop receiving routed tickets. **The frontend should call this when an agent logs out.**

**Response (204 No Content)**

---

## System Architecture Notes for Frontend Developers

1. **Async Processing:** When you `POST /tickets`, the ML analysis (Transformers) happens in the background. You will instantly get a `202 Accepted`. You do not have to wait for the ML model to finish.
2. **Webhooks / Socket Integrations:** (If implemented in M1/M2) High urgency tickets will trigger outbound Webhook POST requests to the URL configured in `.env` (`WEBHOOK_URL`). 
3. **Categories:** The ML Model (M1 and M2) predicts three primary categories: `Billing`, `Technical`, and `Legal`. Ensure your agent skill profiles map exactly to these strings.
4. **Urgency Scores:** The Multi-task Transformer outputs an Urgency Score (0.0 to 1.0). The Orchestrator automatically treats scores > 0.8 as `HIGH` urgency, which heavily influences the Skill-Based Routing logic.
