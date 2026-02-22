// =============================================================================
// Smart-Support API — TypeScript Interfaces
// =============================================================================

export interface HealthResponse {
    status: 'ok' | 'degraded' | 'error';
    queue_size: number;
    classifier: string;
    redis?: 'connected' | 'disconnected';
    app_version?: string;
    circuit_breaker_state?: 'CLOSED' | 'OPEN' | 'HALF_OPEN';
    agents_online?: number;
}

// ── Tickets ──────────────────────────────────────────────────────────────────

export interface TicketPayload {
    ticket_id?: string;
    subject: string;
    body: string;
    customer_id?: string;
    channel?: string;
}

export interface TicketResponse {
    ticket_id: string;
    status: 'accepted' | 'duplicate';
    message: string;
    duplicate: boolean;
    // enriched classification fields
    category: string;
    urgency: string;
    urgency_score: number;
    confidence: number;
    queue_position: number;
    classifier_votes: Record<string, number>;
    timestamp: number;
    // routing fields (CSP)
    routed_to: string | null;
    routing_score: number;
    routing_reason: string | null;
}

// ── Incidents ─────────────────────────────────────────────────────────────────

export interface Incident {
    incident_id: string;
    created_at: number; // UNIX timestamp
    similar_count: number;
    sample_text: string;
    status: 'open' | 'closed';
}

export interface IncidentsResponse {
    incidents: Incident[];
    total: number;
}

// ── Agents ───────────────────────────────────────────────────────────────────

export type SkillCategory = 'Billing' | 'Technical' | 'Legal';

export interface AgentSkills {
    Billing: number;   // 0.0 – 1.0
    Technical: number;
    Legal: number;
}

export interface Agent {
    agent_id: string;
    name: string;
    skills: AgentSkills;
    max_capacity: number;
    active_tickets: number;
    assigned_tickets: any[];
}

export interface AgentsResponse {
    agents: Agent[];
    total: number;
}

export interface AgentPayload {
    agent_id: string;
    name: string;
    skills: AgentSkills;
    max_capacity: number;
}

export interface ReleaseResponse {
    agent_id: string;
    active_tickets: number;
}
