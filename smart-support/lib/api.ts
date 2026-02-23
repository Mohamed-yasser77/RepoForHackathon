// =============================================================================
// Smart-Support API Client — Axios wrapper
// =============================================================================
import axios from 'axios';
import type {
    HealthResponse,
    TicketPayload,
    TicketResponse,
    IncidentsResponse,
    Incident,
    AgentsResponse,
    Agent,
    AgentPayload,
    ReleaseResponse,
} from '@/types';

const BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

const api = axios.create({
    baseURL: BASE_URL,
    headers: { 'Content-Type': 'application/json' },
    timeout: 900_000,
});

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthResponse> {
    const { data } = await api.get<HealthResponse>('/health');
    return data;
}

// ── Tickets ───────────────────────────────────────────────────────────────────

export async function submitTicket(payload: TicketPayload): Promise<TicketResponse> {
    const { data } = await api.post<TicketResponse>('/tickets', payload);
    return data;
}

// ── Incidents ─────────────────────────────────────────────────────────────────

export async function getIncidents(): Promise<IncidentsResponse> {
    const { data } = await api.get<IncidentsResponse>('/incidents');
    return data;
}

export async function getIncident(id: string): Promise<Incident> {
    const { data } = await api.get<Incident>(`/incidents/${id}`);
    return data;
}

// ── Agents ────────────────────────────────────────────────────────────────────

export async function getAgents(): Promise<AgentsResponse> {
    const { data } = await api.get<AgentsResponse>('/agents');
    return data;
}

export async function getAgent(agentId: string): Promise<Agent> {
    const { data } = await api.get<Agent>(`/agents/${agentId}`);
    return data;
}

export async function registerAgent(payload: AgentPayload): Promise<Agent> {
    const { data } = await api.post<Agent>('/agents', payload);
    return data;
}

export async function releaseAgent(agentId: string): Promise<ReleaseResponse> {
    const { data } = await api.post<ReleaseResponse>(`/agents/${agentId}/release`);
    return data;
}

export async function deleteAgent(agentId: string): Promise<void> {
    await api.delete(`/agents/${agentId}`);
}

export default api;
