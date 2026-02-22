'use client';
import { useState } from 'react';
import { Plus, Ticket, Send, Info } from 'lucide-react';
import PageWrapper from '@/components/layout/PageWrapper';
import TicketForm from '@/components/ui/TicketForm';

const channels = ['web', 'email', 'phone', 'chat'] as const;

const sampleTickets = [
    { id: 'TKT-001', subject: 'System crash on login', channel: 'email', urgency: 'HIGH', category: 'Technical', time: '2m ago' },
    { id: 'TKT-002', subject: 'Invoice not received for March', channel: 'web', urgency: 'LOW', category: 'Billing', time: '8m ago' },
    { id: 'TKT-003', subject: 'Data privacy violation inquiry', channel: 'phone', urgency: 'MEDIUM', category: 'Legal', time: '15m ago' },
    { id: 'TKT-004', subject: 'API rate limit exceeded', channel: 'email', urgency: 'HIGH', category: 'Technical', time: '31m ago' },
    { id: 'TKT-005', subject: 'Refund request – duplicate charge', channel: 'chat', urgency: 'MEDIUM', category: 'Billing', time: '1h ago' },
];

const urgencyBadge: Record<string, string> = {
    HIGH: 'badge-red',
    MEDIUM: 'badge-orange',
    LOW: 'badge-green',
};

const categoryBadge: Record<string, string> = {
    Technical: 'badge-blue',
    Billing: 'badge-purple',
    Legal: 'badge-orange',
};

export default function TicketsPage() {
    const [showForm, setShowForm] = useState(false);

    return (
        <PageWrapper>
            <div className="page">
                <div className="page-header reveal">
                    <div className="flex items-center" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
                        <div>
                            <h1 className="page-title">Ticket Center</h1>
                            <p className="page-subtitle">Submit and monitor support tickets through the M3 pipeline</p>
                        </div>
                        <button className="btn btn-primary" onClick={() => setShowForm(true)}>
                            <Plus size={16} /> New Ticket
                        </button>
                    </div>
                </div>

                {/* Pipeline info */}
                <div className="card reveal" style={{ marginBottom: 24 }}>
                    <div className="flex items-center gap-3" style={{ flexWrap: 'wrap' }}>
                        <Info size={16} color="var(--accent-blue)" />
                        <span className="text-sm text-secondary">
                            Submitted tickets are <strong style={{ color: 'var(--text-primary)' }}>immediately accepted (202)</strong> and queued for async ML processing. The pipeline runs:
                        </span>
                        {['Redlock dedupe', 'Semantic similarity', 'Circuit Breaker check', 'Transformer model', 'Skill routing'].map((s, i) => (
                            <span key={s} className="flex items-center gap-1">
                                {i > 0 && <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>→</span>}
                                <span className="badge badge-green" style={{ fontSize: 10 }}>{s}</span>
                            </span>
                        ))}
                    </div>
                </div>

                {/* API Endpoint quick reference */}
                <div className="grid-2 reveal" style={{ marginBottom: 24 }}>
                    <div className="card">
                        <h3 className="section-title"><Send size={14} color="var(--accent-green)" /> POST /tickets</h3>
                        <p className="text-sm text-secondary" style={{ lineHeight: 1.6, marginBottom: 14 }}>
                            Submit a ticket for async processing. Returns <span className="badge badge-green" style={{ fontSize: 10 }}>202 ACCEPTED</span> instantly.
                        </p>
                        <div style={{ padding: '12px 14px', background: 'rgba(0,255,178,0.04)', borderRadius: 8, border: '1px solid rgba(0,255,178,0.1)' }}>
                            <pre className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)', overflow: 'auto', whiteSpace: 'pre-wrap' }}>{`{
  "ticket_id": "TKT-001",
  "subject": "System crash",
  "body": "Detailed description...",
  "customer_id": "CUST-123",  // optional
  "channel": "email"          // optional
}`}</pre>
                        </div>
                    </div>

                    <div className="card card-purple">
                        <h3 className="section-title"><Ticket size={14} color="var(--accent-purple)" /> Response Shape</h3>
                        <p className="text-sm text-secondary" style={{ lineHeight: 1.6, marginBottom: 14 }}>
                            The <code className="font-mono" style={{ fontSize: 11, color: 'var(--accent-purple)' }}>duplicate</code> flag is set by <strong>Redlock</strong> when the same ticket_id is submitted within 5 seconds.
                        </p>
                        <div style={{ padding: '12px 14px', background: 'rgba(123,97,255,0.04)', borderRadius: 8, border: '1px solid rgba(123,97,255,0.1)' }}>
                            <pre className="font-mono" style={{ fontSize: 11, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>{`{
  "ticket_id": "TKT-001",
  "status": "accepted",
  "message": "Enqueued for M3 pipeline.",
  "duplicate": false
}`}</pre>
                        </div>
                    </div>
                </div>

                {/* Recent tickets table */}
                <div className="card reveal">
                    <h3 className="section-title">
                        <Ticket size={16} color="var(--accent-green)" />
                        Recent Tickets
                        <span className="badge badge-gray" style={{ marginLeft: 'auto' }}>Demo Data</span>
                    </h3>
                    <table className="data-table w-full">
                        <thead>
                            <tr>
                                <th>Ticket ID</th>
                                <th>Subject</th>
                                <th>Channel</th>
                                <th>Category</th>
                                <th>Urgency</th>
                                <th>Submitted</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sampleTickets.map((t) => (
                                <tr key={t.id}>
                                    <td className="font-mono text-xs" style={{ color: 'var(--accent-green)' }}>{t.id}</td>
                                    <td>{t.subject}</td>
                                    <td><span className="badge badge-gray">{t.channel}</span></td>
                                    <td><span className={`badge ${categoryBadge[t.category]}`}>{t.category}</span></td>
                                    <td><span className={`badge ${urgencyBadge[t.urgency]}`}>{t.urgency}</span></td>
                                    <td className="text-muted text-xs">{t.time}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {showForm && <TicketForm onClose={() => setShowForm(false)} />}
        </PageWrapper>
    );
}
