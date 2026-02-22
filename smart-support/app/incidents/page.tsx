'use client';
import { AlertTriangle, RefreshCw, CheckCircle, Flame } from 'lucide-react';
import PageWrapper from '@/components/layout/PageWrapper';
import IncidentCard from '@/components/ui/IncidentCard';
import StatCounter from '@/components/ui/StatCounter';
import { useIncidents } from '@/lib/hooks';

export default function IncidentsPage() {
    const { data, loading, error, refetch } = useIncidents(10_000);

    const total = data?.total ?? 0;
    const openCount = data?.incidents.filter((i) => i.status === 'open').length ?? 0;
    const maxSimilar = data?.incidents.reduce((m, i) => Math.max(m, i.similar_count), 0) ?? 0;

    return (
        <PageWrapper>
            <div className="page">
                <div className="page-header reveal">
                    <div className="flex items-center" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
                        <div>
                            <h1 className="page-title">Flash-Flood Monitor</h1>
                            <p className="page-subtitle">
                                Active Master Incidents detected by the Semantic Deduplicator · auto-refreshes every 10s
                            </p>
                        </div>
                        <button className="btn btn-ghost btn-sm" onClick={refetch}>
                            <RefreshCw size={13} /> Refresh
                        </button>
                    </div>
                </div>

                {/* Stats */}
                <div className="grid-3 reveal" style={{ marginBottom: 24 }}>
                    <div className="card card-orange">
                        <div className="flex items-center gap-2 mb-4">
                            <Flame size={14} color="var(--accent-orange)" />
                            <span className="text-xs text-muted" style={{ letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>Total Incidents</span>
                        </div>
                        <StatCounter value={total} label="Master incidents detected" color="orange" />
                    </div>

                    <div className="card">
                        <div className="flex items-center gap-2 mb-4">
                            <AlertTriangle size={14} color="var(--accent-green)" />
                            <span className="text-xs text-muted" style={{ letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>Open Incidents</span>
                        </div>
                        <StatCounter value={openCount} label="Currently active" color="green" />
                    </div>

                    <div className="card card-purple">
                        <div className="flex items-center gap-2 mb-4">
                            <CheckCircle size={14} color="var(--accent-purple)" />
                            <span className="text-xs text-muted" style={{ letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>Peak Similarity Group</span>
                        </div>
                        <StatCounter value={maxSimilar} label="Max tickets in one incident" color="purple" />
                    </div>
                </div>

                {/* How it works */}
                <div className="card reveal" style={{ marginBottom: 24 }}>
                    <div className="flex" style={{ gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                        <div style={{
                            width: 36, height: 36, borderRadius: 8, background: 'var(--accent-orange-dim)', border: '1px solid rgba(255,107,53,0.2)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0
                        }}>
                            <Flame size={18} color="var(--accent-orange)" />
                        </div>
                        <div>
                            <div className="font-semibold" style={{ color: 'var(--text-primary)', marginBottom: 6 }}>Flash-Flood Detection Algorithm</div>
                            <p className="text-sm text-secondary" style={{ lineHeight: 1.7 }}>
                                When <strong style={{ color: 'var(--text-primary)' }}>&gt;10 tickets</strong> with{' '}
                                <span className="badge badge-orange" style={{ fontSize: 10 }}>Cosine Similarity &gt; 0.9</span> are received within{' '}
                                <strong style={{ color: 'var(--text-primary)' }}>5 minutes</strong>, the Semantic Deduplicator groups them into a single Master Incident to prevent alert spam.
                                The incident gets an auto-generated hash ID and all subsequent similar tickets are folded in.
                            </p>
                        </div>
                    </div>
                </div>

                {/* Incidents grid */}
                {loading && (
                    <div className="grid-3">
                        {Array.from({ length: 3 }).map((_, i) => (
                            <div key={i} className="skeleton" style={{ height: 200, borderRadius: 20 }} />
                        ))}
                    </div>
                )}

                {error && (
                    <div className="card" style={{ textAlign: 'center', padding: 48 }}>
                        <AlertTriangle size={32} color="var(--accent-orange)" style={{ margin: '0 auto 12px' }} />
                        <p className="text-secondary">{error}</p>
                        <button className="btn btn-ghost" style={{ marginTop: 16 }} onClick={refetch}>Retry</button>
                    </div>
                )}

                {!loading && !error && (data?.incidents.length === 0 ? (
                    <div className="card" style={{ textAlign: 'center', padding: 64 }}>
                        <CheckCircle size={40} color="var(--accent-green)" style={{ margin: '0 auto 12px' }} />
                        <h3 style={{ fontWeight: 700, fontSize: 18, marginBottom: 8 }}>No Active Incidents</h3>
                        <p className="text-secondary">The system is operating normally — no flash-flood patterns detected.</p>
                    </div>
                ) : (
                    <div className="grid-3">
                        {data?.incidents.map((inc) => (
                            <IncidentCard key={inc.incident_id} incident={inc} />
                        ))}
                    </div>
                ))}
            </div>
        </PageWrapper>
    );
}
