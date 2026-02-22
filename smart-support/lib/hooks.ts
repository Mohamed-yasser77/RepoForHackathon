'use client';
// =============================================================================
// Custom React Hooks for Smart-Support API
// =============================================================================
import { useState, useEffect, useCallback, useRef } from 'react';
import { getHealth, getIncidents, getAgents, submitTicket, registerAgent, releaseAgent, deleteAgent } from './api';
import type { HealthResponse, IncidentsResponse, AgentsResponse, TicketPayload, AgentPayload } from '@/types';

// ── Generic polling hook ──────────────────────────────────────────────────────

function usePolling<T>(
    fetcher: () => Promise<T>,
    interval = 10_000
) {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetch = useCallback(async () => {
        try {
            const result = await fetcher();
            setData(result);
            setError(null);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    }, [fetcher]);

    useEffect(() => {
        fetch();
        const id = setInterval(fetch, interval);
        return () => clearInterval(id);
    }, [fetch, interval]);

    return { data, loading, error, refetch: fetch };
}

// ── Health ────────────────────────────────────────────────────────────────────

export function useHealth(interval = 8_000) {
    const fetcher = useCallback(() => getHealth(), []);
    return usePolling<HealthResponse>(fetcher, interval);
}

// ── Incidents ─────────────────────────────────────────────────────────────────

export function useIncidents(interval = 10_000) {
    const fetcher = useCallback(() => getIncidents(), []);
    return usePolling<IncidentsResponse>(fetcher, interval);
}

// ── Agents ────────────────────────────────────────────────────────────────────

export function useAgents(interval = 8_000) {
    const fetcher = useCallback(() => getAgents(), []);
    return usePolling<AgentsResponse>(fetcher, interval);
}

// ── Submit Ticket ─────────────────────────────────────────────────────────────

export function useSubmitTicket() {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<{ success: boolean; duplicate?: boolean; message?: string } | null>(null);
    const [error, setError] = useState<string | null>(null);

    const submit = useCallback(async (payload: TicketPayload) => {
        setLoading(true);
        setError(null);
        try {
            const res = await submitTicket(payload);
            setResult({ success: true, duplicate: res.duplicate, message: res.message });
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : 'Failed to submit ticket');
            setResult(null);
        } finally {
            setLoading(false);
        }
    }, []);

    return { submit, loading, result, error };
}

// ── Register Agent ────────────────────────────────────────────────────────────

export function useRegisterAgent() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const register = useCallback(async (payload: AgentPayload) => {
        setLoading(true);
        setError(null);
        try {
            await registerAgent(payload);
            return true;
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : 'Failed to register agent');
            return false;
        } finally {
            setLoading(false);
        }
    }, []);

    return { register, loading, error };
}

// ── Release / Delete Agent ────────────────────────────────────────────────────

export function useAgentActions() {
    const [loading, setLoading] = useState<string | null>(null);

    const release = useCallback(async (agentId: string) => {
        setLoading(agentId);
        try { await releaseAgent(agentId); } finally { setLoading(null); }
    }, []);

    const remove = useCallback(async (agentId: string) => {
        setLoading(agentId);
        try { await deleteAgent(agentId); } finally { setLoading(null); }
    }, []);

    return { release, remove, loading };
}
