'use client';
import { AlertTriangle, Clock, Copy } from 'lucide-react';
import type { Incident } from '@/types';

interface IncidentCardProps {
    incident: Incident;
}

function formatTime(unix: number): string {
    return new Date(unix * 1000).toLocaleString();
}

export default function IncidentCard({ incident }: IncidentCardProps) {
    const severity = incident.similar_count >= 50 ? 'critical' : incident.similar_count >= 20 ? 'high' : 'medium';
    const severityConfig = {
        critical: { cls: 'badge-red', label: 'CRITICAL' },
        high: { cls: 'badge-orange', label: 'HIGH' },
        medium: { cls: 'badge-purple', label: 'MEDIUM' },
    }[severity];

    return (
        <div className="card card-orange">
            {/* Header */}
            <div className="flex items-center" style={{ justifyContent: 'space-between', marginBottom: 14 }}>
                <div className="flex items-center gap-2">
                    <AlertTriangle size={16} color="var(--accent-orange)" />
                    <span className="font-mono text-xs text-muted">{incident.incident_id}</span>
                </div>
                <span className={`badge ${severityConfig.cls}`}>
                    <span className={`dot ${severity === 'critical' ? 'dot-red' : 'dot-orange'}`} />
                    {severityConfig.label}
                </span>
            </div>

            {/* Sample text */}
            <p style={{
                fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.6,
                marginBottom: 16, display: '-webkit-box',
                WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden'
            }}>
                {incident.sample_text}
            </p>

            {/* Stats row */}
            <div className="flex gap-4" style={{ flexWrap: 'wrap' }}>
                <div style={{ textAlign: 'center', flex: 1, minWidth: 80 }}>
                    <div style={{
                        fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 700,
                        color: 'var(--accent-orange)', textShadow: '0 0 20px rgba(255,107,53,0.4)'
                    }}>
                        {incident.similar_count}
                    </div>
                    <div className="text-xs text-muted" style={{ marginTop: 2 }}>Similar Tickets</div>
                </div>
                <div style={{ width: '1px', background: 'var(--border-subtle)' }} />
                <div style={{ flex: 2 }}>
                    <div className="flex items-center gap-2 mb-2">
                        <Clock size={12} color="var(--text-muted)" />
                        <span className="text-xs text-muted">Detected</span>
                    </div>
                    <div className="font-mono text-xs" style={{ color: 'var(--text-secondary)' }}>
                        {formatTime(incident.created_at)}
                    </div>
                    <div className="mt-2">
                        <span className={`badge ${incident.status === 'open' ? 'badge-orange' : 'badge-gray'}`}>
                            {incident.status.toUpperCase()}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
