'use client';
import { User, Briefcase, Trash2, CheckCircle, Loader } from 'lucide-react';
import SkillBar from './SkillBar';
import type { Agent } from '@/types';

interface AgentCardProps {
    agent: Agent;
    onRelease?: (id: string) => void;
    onDelete?: (id: string) => void;
    actionLoading?: string | null; // agentId being acted on
}

const skillColors: Record<string, 'green' | 'purple' | 'orange'> = {
    Technical: 'green',
    Billing: 'purple',
    Legal: 'orange',
};

export default function AgentCard({ agent, onRelease, onDelete, actionLoading }: AgentCardProps) {
    const loadPct = agent.max_capacity > 0 ? agent.active_tickets / agent.max_capacity : 0;
    const loadColor = loadPct >= 1 ? '#FF5B5B' : loadPct >= 0.7 ? 'var(--accent-orange)' : 'var(--accent-green)';
    const isLoading = actionLoading === agent.agent_id;

    return (
        <div className="card card-purple reveal" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Header */}
            <div className="flex items-center gap-3" style={{ justifyContent: 'space-between' }}>
                <div className="flex items-center gap-3">
                    <div style={{
                        width: 40, height: 40, borderRadius: '50%',
                        background: 'linear-gradient(135deg, var(--accent-purple), var(--accent-green))',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        flexShrink: 0
                    }}>
                        <User size={18} color="#080A10" />
                    </div>
                    <div>
                        <div className="font-semibold" style={{ color: 'var(--text-primary)', fontSize: 15 }}>{agent.name}</div>
                        <div className="font-mono text-xs text-muted">{agent.agent_id}</div>
                    </div>
                </div>
                <div style={{
                    fontSize: 11, fontFamily: 'var(--font-mono)', fontWeight: 600,
                    color: loadColor, padding: '3px 10px',
                    borderRadius: 999, border: `1px solid ${loadColor}22`,
                    background: `${loadColor}18`,
                }}>
                    {agent.active_tickets}/{agent.max_capacity}
                </div>
            </div>

            {/* Load bar */}
            <div>
                <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-muted font-semibold" style={{ letterSpacing: '0.05em', textTransform: 'uppercase' }}>Ticket Load</span>
                    <span className="text-xs font-mono text-muted">{Math.round(loadPct * 100)}%</span>
                </div>
                <div className="skill-bar-track">
                    <div
                        className="skill-bar-fill"
                        style={{
                            width: `${loadPct * 100}%`,
                            background: `linear-gradient(90deg, ${loadColor}, ${loadColor}bb)`,
                            boxShadow: `0 0 8px ${loadColor}44`,
                            transition: 'width 1s ease'
                        }}
                    />
                </div>
            </div>

            {/* Skills */}
            <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 14 }}>
                <div className="flex items-center gap-2 mb-3">
                    <Briefcase size={12} color="var(--text-muted)" />
                    <span className="text-xs text-muted font-semibold" style={{ letterSpacing: '0.05em', textTransform: 'uppercase' }}>Skills</span>
                </div>
                {Object.entries(agent.skills).map(([skill, val], i) => (
                    <SkillBar
                        key={skill}
                        label={skill}
                        value={val}
                        color={skillColors[skill] ?? 'green'}
                        delay={i * 0.1}
                    />
                ))}
            </div>

            {/* Actions */}
            {(onRelease || onDelete) && (
                <div className="flex gap-2 mt-2" style={{ justifyContent: 'flex-end' }}>
                    {onRelease && (
                        <button
                            className="btn btn-purple btn-sm"
                            onClick={() => onRelease(agent.agent_id)}
                            disabled={!!isLoading || agent.active_tickets <= 0}
                        >
                            {isLoading ? <Loader size={12} className="spin" /> : <CheckCircle size={12} />}
                            Release
                        </button>
                    )}
                    {onDelete && (
                        <button
                            className="btn btn-danger btn-sm"
                            onClick={() => onDelete(agent.agent_id)}
                            disabled={!!isLoading}
                        >
                            {isLoading ? <Loader size={12} className="spin" /> : <Trash2 size={12} />}
                            Remove
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}
