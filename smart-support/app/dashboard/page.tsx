'use client';
import { Activity, Wifi, WifiOff, Shield, Users, RefreshCw } from 'lucide-react';
import PageWrapper from '@/components/layout/PageWrapper';
import CircuitBreakerBadge from '@/components/ui/CircuitBreakerBadge';
import StatCounter from '@/components/ui/StatCounter';
import { useHealth, useAgents } from '@/lib/hooks';

export default function DashboardPage() {
    const { data: health, loading: hLoading, error: hError, refetch } = useHealth(8000);
    const { data: agents } = useAgents(8000);

    const totalLoad = agents?.agents.reduce((s, a) => s + a.active_tickets, 0) ?? 0;
    const totalCapacity = agents?.agents.reduce((s, a) => s + a.max_capacity, 0) ?? 0;
    const loadPct = totalCapacity > 0 ? ((totalLoad / totalCapacity) * 100).toFixed(1) : '0';

    return (
        <PageWrapper>
            <div className="page">
                <div className="page-header reveal">
                    <div className="flex items-center gap-3" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
                        <div>
                            <h1 className="page-title">System Dashboard</h1>
                            <p className="page-subtitle">Real-time health monitoring and orchestrator status</p>
                        </div>
                        <button className="btn btn-ghost btn-sm" onClick={refetch}>
                            <RefreshCw size={13} /> Refresh
                        </button>
                    </div>
                </div>

                {/* Circuit Breaker Banner */}
                <div className="reveal" style={{ marginBottom: 24 }}>
                    {hLoading ? (
                        <div className="skeleton" style={{ height: 48, borderRadius: 12 }} />
                    ) : health ? (
                        <CircuitBreakerBadge state={health.circuit_breaker_state} />
                    ) : (
                        <div className="circuit-breaker" style={{ borderColor: 'rgba(255,91,91,0.3)', background: 'rgba(255,50,50,0.1)', color: '#FF5B5B' }}>
                            <WifiOff size={14} /> Cannot reach API — {hError}
                        </div>
                    )}
                </div>

                {/* Stats grid */}
                <div className="grid-4 reveal" style={{ marginBottom: 24 }}>
                    {/* Status */}
                    <div className="card">
                        <div className="flex items-center gap-2 mb-4">
                            <Activity size={14} color="var(--accent-green)" />
                            <span className="text-xs text-muted" style={{ letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>API Status</span>
                        </div>
                        {hLoading ? (
                            <div className="skeleton" style={{ height: 40, borderRadius: 8 }} />
                        ) : (
                            <>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                                    <span className={`dot ${health?.status === 'ok' ? 'dot-green' : 'dot-red'}`} />
                                    <span style={{
                                        fontFamily: 'var(--font-mono)', fontSize: 24, fontWeight: 700,
                                        color: health?.status === 'ok' ? 'var(--accent-green)' : '#FF5B5B',
                                        textTransform: 'uppercase'
                                    }}>
                                        {health?.status ?? '—'}
                                    </span>
                                </div>
                                <p className="text-sm text-secondary">System operational</p>
                            </>
                        )}
                    </div>

                    {/* Redis */}
                    <div className="card card-purple">
                        <div className="flex items-center gap-2 mb-4">
                            {health?.redis === 'connected' ? <Wifi size={14} color="var(--accent-purple)" /> : <WifiOff size={14} color="#FF5B5B" />}
                            <span className="text-xs text-muted" style={{ letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>Redis</span>
                        </div>
                        {hLoading ? (
                            <div className="skeleton" style={{ height: 40, borderRadius: 8 }} />
                        ) : (
                            <>
                                <div style={{
                                    fontFamily: 'var(--font-mono)', fontSize: 20, fontWeight: 700,
                                    color: health?.redis === 'connected' ? 'var(--accent-purple)' : '#FF5B5B',
                                    textTransform: 'uppercase', marginBottom: 4
                                }}>
                                    {health?.redis ?? '—'}
                                </div>
                                <p className="text-sm text-secondary">Cache layer</p>
                            </>
                        )}
                    </div>

                    {/* Agents online */}
                    <div className="card">
                        <div className="flex items-center gap-2 mb-4">
                            <Users size={14} color="var(--accent-green)" />
                            <span className="text-xs text-muted" style={{ letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>Agents Online</span>
                        </div>
                        {hLoading ? (
                            <div className="skeleton" style={{ height: 40, borderRadius: 8 }} />
                        ) : (
                            <StatCounter value={health?.agents_online ?? 0} label="Active in routing pool" color="green" />
                        )}
                    </div>

                    {/* Version */}
                    <div className="card card-purple">
                        <div className="flex items-center gap-2 mb-4">
                            <Shield size={14} color="var(--accent-purple)" />
                            <span className="text-xs text-muted" style={{ letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>Version</span>
                        </div>
                        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 700, color: 'var(--accent-purple)', marginBottom: 4 }}>
                            {health?.app_version ?? '—'}
                        </div>
                        <p className="text-sm text-secondary">M3 Pipeline</p>
                    </div>
                </div>

                {/* Agent load overview */}
                <div className="grid-2 reveal">
                    <div className="card">
                        <h3 className="section-title">
                            <Users size={16} color="var(--accent-green)" />
                            Agent Pool Load
                        </h3>
                        <div className="flex items-center gap-4" style={{ marginBottom: 16 }}>
                            <StatCounter value={totalLoad} label="Active tickets total" color="green" />
                            <div style={{ width: 1, height: 50, background: 'var(--border-subtle)' }} />
                            <StatCounter value={totalCapacity} label="Total capacity" color="purple" />
                            <div style={{ width: 1, height: 50, background: 'var(--border-subtle)' }} />
                            <div>
                                <div style={{
                                    fontFamily: 'var(--font-mono)', fontSize: 36, fontWeight: 700,
                                    color: Number(loadPct) > 80 ? 'var(--accent-orange)' : 'var(--accent-green)',
                                    textShadow: '0 0 20px currentColor'
                                }}>
                                    {loadPct}%
                                </div>
                                <p className="text-sm text-secondary" style={{ marginTop: 4 }}>Load utilization</p>
                            </div>
                        </div>
                        <div className="skill-bar-track">
                            <div className="skill-bar-fill skill-bar-green" style={{ width: `${loadPct}%`, transition: 'width 1s ease' }} />
                        </div>
                    </div>

                    <div className="card card-purple">
                        <h3 className="section-title" style={{ color: 'var(--accent-purple)' }}>
                            <Activity size={16} />
                            System Notes
                        </h3>
                        {[
                            'Tickets are processed asynchronously via ARQ',
                            'ML Categories: Billing · Technical · Legal',
                            'Urgency > 0.8 triggers HIGH priority routing',
                            'Redlock detects duplicates within 5-second window',
                            'Circuit Breaker: CLOSED = nominal, OPEN = tripped',
                        ].map((note) => (
                            <div key={note} className="flex items-center gap-2" style={{ marginBottom: 10 }}>
                                <div style={{ width: 4, height: 4, borderRadius: '50%', background: 'var(--accent-purple)', flexShrink: 0 }} />
                                <span className="text-sm text-secondary">{note}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </PageWrapper>
    );
}
