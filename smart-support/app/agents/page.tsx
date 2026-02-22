'use client';
import { useState } from 'react';
import { Users, Plus, X, Loader, RefreshCw, CheckCircle, AlertTriangle, User } from 'lucide-react';
import PageWrapper from '@/components/layout/PageWrapper';
import AgentCard from '@/components/ui/AgentCard';
import StatCounter from '@/components/ui/StatCounter';
import { useAgents, useRegisterAgent, useAgentActions } from '@/lib/hooks';
import type { AgentPayload } from '@/types';

const defaultForm: AgentPayload = {
    agent_id: '',
    name: '',
    skills: { Billing: 0.5, Technical: 0.5, Legal: 0.5 },
    max_capacity: 5,
};

export default function AgentsPage() {
    const { data, loading, error, refetch } = useAgents(8000);
    const { register, loading: regLoading, error: regError } = useRegisterAgent();
    const { release, remove, loading: actionLoading } = useAgentActions();

    const [showForm, setShowForm] = useState(false);
    const [form, setForm] = useState<AgentPayload>(defaultForm);
    const [success, setSuccess] = useState(false);

    const totalAgents = data?.total ?? 0;
    const onlineCount = data?.agents.filter((a) => a.active_tickets < a.max_capacity).length ?? 0;
    const atCapacity = data?.agents.filter((a) => a.active_tickets >= a.max_capacity).length ?? 0;

    const handleSkillChange = (skill: keyof typeof form.skills, val: string) => {
        setForm(f => ({ ...f, skills: { ...f.skills, [skill]: parseFloat(val) } }));
    };

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        const ok = await register(form);
        if (ok) {
            setSuccess(true);
            setForm(defaultForm);
            refetch();
            setTimeout(() => { setSuccess(false); setShowForm(false); }, 2000);
        }
    };

    const handleRelease = async (id: string) => {
        await release(id);
        refetch();
    };
    const handleDelete = async (id: string) => {
        await remove(id);
        refetch();
    };

    return (
        <PageWrapper>
            <div className="page">
                <div className="page-header reveal">
                    <div className="flex items-center" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
                        <div>
                            <h1 className="page-title">Agent Roster</h1>
                            <p className="page-subtitle">Register, monitor, and manage skill-based routing agents</p>
                        </div>
                        <div className="flex gap-2">
                            <button className="btn btn-ghost btn-sm" onClick={refetch}>
                                <RefreshCw size={13} /> Refresh
                            </button>
                            <button className="btn btn-primary" onClick={() => setShowForm(true)}>
                                <Plus size={16} /> Register Agent
                            </button>
                        </div>
                    </div>
                </div>

                {/* Stats */}
                <div className="grid-3 reveal" style={{ marginBottom: 24 }}>
                    <div className="card">
                        <div className="flex items-center gap-2 mb-4">
                            <Users size={14} color="var(--accent-green)" />
                            <span className="text-xs text-muted" style={{ letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>Total Agents</span>
                        </div>
                        <StatCounter value={totalAgents} label="In routing pool" color="green" />
                    </div>

                    <div className="card">
                        <div className="flex items-center gap-2 mb-4">
                            <CheckCircle size={14} color="var(--accent-green)" />
                            <span className="text-xs text-muted" style={{ letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>Available</span>
                        </div>
                        <StatCounter value={onlineCount} label="Below max capacity" color="green" />
                    </div>

                    <div className="card card-orange">
                        <div className="flex items-center gap-2 mb-4">
                            <AlertTriangle size={14} color="var(--accent-orange)" />
                            <span className="text-xs text-muted" style={{ letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600 }}>At Capacity</span>
                        </div>
                        <StatCounter value={atCapacity} label="Fully loaded agents" color="orange" />
                    </div>
                </div>

                {/* Routing info */}
                <div className="card card-purple reveal" style={{ marginBottom: 24 }}>
                    <h3 className="section-title" style={{ color: 'var(--accent-purple)', marginBottom: 12 }}>
                        <Users size={14} /> Skill-Based Routing — How It Works
                    </h3>
                    <div className="grid-3">
                        {[
                            { label: 'Billing', color: 'var(--accent-purple)', desc: 'Handles payment, invoice, refund disputes' },
                            { label: 'Technical', color: 'var(--accent-green)', desc: 'API, crashes, integration failures' },
                            { label: 'Legal', color: 'var(--accent-orange)', desc: 'Privacy, compliance, regulatory inquiries' },
                        ].map(({ label, color, desc }) => (
                            <div key={label} style={{ padding: '12px 16px', background: 'rgba(255,255,255,0.03)', borderRadius: 12, border: '1px solid var(--border-subtle)' }}>
                                <div style={{ fontWeight: 700, color, marginBottom: 4, fontSize: 13 }}>{label}</div>
                                <p className="text-xs text-secondary" style={{ lineHeight: 1.5 }}>{desc}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Agents grid */}
                {loading && (
                    <div className="grid-3">
                        {Array.from({ length: 3 }).map((_, i) => (
                            <div key={i} className="skeleton" style={{ height: 280, borderRadius: 20 }} />
                        ))}
                    </div>
                )}

                {error && (
                    <div className="card" style={{ textAlign: 'center', padding: 48 }}>
                        <AlertTriangle size={32} color="var(--accent-orange)" style={{ margin: '0 auto 12px' }} />
                        <p className="text-secondary">{error}</p>
                    </div>
                )}

                {!loading && !error && (data?.agents.length === 0 ? (
                    <div className="card" style={{ textAlign: 'center', padding: 64 }}>
                        <User size={40} color="var(--text-muted)" style={{ margin: '0 auto 12px' }} />
                        <h3 style={{ fontWeight: 700, fontSize: 18, marginBottom: 8 }}>No Agents Registered</h3>
                        <p className="text-secondary" style={{ marginBottom: 20 }}>Register an agent to start routing tickets.</p>
                        <button className="btn btn-primary" onClick={() => setShowForm(true)}><Plus size={16} /> Register First Agent</button>
                    </div>
                ) : (
                    <div className="grid-3">
                        {data?.agents.map((agent) => (
                            <AgentCard
                                key={agent.agent_id}
                                agent={agent}
                                onRelease={handleRelease}
                                onDelete={handleDelete}
                                actionLoading={actionLoading}
                            />
                        ))}
                    </div>
                ))}
            </div>

            {/* Register Agent Modal */}
            {showForm && (
                <div className="modal-backdrop" onClick={(e) => { if (e.target === e.currentTarget) setShowForm(false); }}>
                    <div className="modal">
                        <button className="modal-close" onClick={() => setShowForm(false)}><X size={18} /></button>
                        <div className="modal-title">
                            <div style={{ width: 32, height: 32, borderRadius: 8, background: 'var(--accent-purple-dim)', border: '1px solid rgba(123,97,255,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <User size={16} color="var(--accent-purple)" />
                            </div>
                            Register New Agent
                        </div>

                        {success ? (
                            <div style={{ textAlign: 'center', padding: '24px 0' }}>
                                <CheckCircle size={40} color="var(--accent-green)" style={{ margin: '0 auto 12px' }} />
                                <div style={{ fontWeight: 700, fontSize: 18, color: 'var(--accent-green)' }}>Agent Registered!</div>
                            </div>
                        ) : (
                            <form onSubmit={handleRegister}>
                                <div className="form-row mb-4">
                                    <div className="form-group" style={{ marginBottom: 0 }}>
                                        <label className="label">Agent ID *</label>
                                        <input className="input font-mono" style={{ fontSize: 12 }} placeholder="alice_tech_01" required
                                            value={form.agent_id} onChange={(e) => setForm(f => ({ ...f, agent_id: e.target.value }))} />
                                    </div>
                                    <div className="form-group" style={{ marginBottom: 0 }}>
                                        <label className="label">Display Name *</label>
                                        <input className="input" placeholder="Alice" required
                                            value={form.name} onChange={(e) => setForm(f => ({ ...f, name: e.target.value }))} />
                                    </div>
                                </div>

                                <div className="form-group">
                                    <label className="label">Max Capacity</label>
                                    <input type="number" className="input" min={1} max={20}
                                        value={form.max_capacity} onChange={(e) => setForm(f => ({ ...f, max_capacity: parseInt(e.target.value) }))} />
                                </div>

                                <div className="form-group">
                                    <label className="label">Skill Scores (0.0 – 1.0)</label>
                                    {(['Billing', 'Technical', 'Legal'] as const).map((skill, i) => {
                                        const colors = ['purple', 'green', 'orange'] as const;
                                        const accent = ['var(--accent-purple)', 'var(--accent-green)', 'var(--accent-orange)'][i];
                                        return (
                                            <div key={skill} style={{ marginBottom: 12 }}>
                                                <div className="flex justify-between items-center mb-1">
                                                    <label className="label" style={{ margin: 0, color: accent }}>{skill}</label>
                                                    <span className="font-mono text-xs" style={{ color: accent }}>{form.skills[skill].toFixed(2)}</span>
                                                </div>
                                                <input type="range" min={0} max={1} step={0.05}
                                                    value={form.skills[skill]}
                                                    onChange={(e) => handleSkillChange(skill, e.target.value)}
                                                    style={{ width: '100%', accentColor: accent, cursor: 'pointer' }}
                                                />
                                            </div>
                                        );
                                    })}
                                </div>

                                {regError && (
                                    <div className="badge badge-red" style={{ marginBottom: 16, width: '100%', justifyContent: 'flex-start' }}>
                                        <AlertTriangle size={12} /> {regError}
                                    </div>
                                )}

                                <div className="flex gap-3" style={{ justifyContent: 'flex-end' }}>
                                    <button type="button" className="btn btn-ghost" onClick={() => setShowForm(false)}>Cancel</button>
                                    <button type="submit" className="btn btn-primary" disabled={regLoading}>
                                        {regLoading ? <><Loader size={14} className="spin" /> Registering…</> : <><User size={14} /> Register Agent</>}
                                    </button>
                                </div>
                            </form>
                        )}
                    </div>
                </div>
            )}
        </PageWrapper>
    );
}
