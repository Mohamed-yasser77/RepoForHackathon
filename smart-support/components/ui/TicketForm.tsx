'use client';
import { useState } from 'react';
import { X, Ticket, Loader, CheckCircle, AlertTriangle } from 'lucide-react';
import { useSubmitTicket } from '@/lib/hooks';

interface TicketFormProps {
    onClose: () => void;
}

export default function TicketForm({ onClose }: TicketFormProps) {
    const { submit, loading, result, error } = useSubmitTicket();
    const [form, setForm] = useState({
        ticket_id: `TKT-${Date.now()}`,
        subject: '',
        body: '',
        customer_id: '',
        channel: 'web',
    });

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        setForm(f => ({ ...f, [e.target.name]: e.target.value }));
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        await submit({
            ticket_id: form.ticket_id,
            subject: form.subject,
            body: form.body,
            customer_id: form.customer_id || undefined,
            channel: form.channel || undefined,
        });
    };

    return (
        <div className="modal-backdrop" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
            <div className="modal">
                <button className="modal-close" onClick={onClose} aria-label="Close">
                    <X size={18} />
                </button>

                <div className="modal-title">
                    <div style={{
                        width: 32, height: 32, borderRadius: 8,
                        background: 'var(--accent-green-dim)', border: '1px solid rgba(0,255,178,0.2)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}>
                        <Ticket size={16} color="var(--accent-green)" />
                    </div>
                    Submit Support Ticket
                </div>

                {result ? (
                    <div style={{ textAlign: 'center', padding: '24px 0' }}>
                        <div style={{ marginBottom: 16 }}>
                            {result.duplicate
                                ? <AlertTriangle size={40} color="var(--accent-orange)" style={{ margin: '0 auto' }} />
                                : <CheckCircle size={40} color="var(--accent-green)" style={{ margin: '0 auto' }} />
                            }
                        </div>
                        <div style={{
                            fontWeight: 700, fontSize: 18, marginBottom: 8,
                            color: result.duplicate ? 'var(--accent-orange)' : 'var(--accent-green)'
                        }}>
                            {result.duplicate ? 'Duplicate Detected' : 'Ticket Accepted'}
                        </div>
                        <p style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.6 }}>
                            {result.message}
                        </p>
                        {result.duplicate && (
                            <div className="badge badge-orange" style={{ marginTop: 12, display: 'inline-flex' }}>
                                Redlock: Duplicate within 5s window
                            </div>
                        )}
                        <div style={{ marginTop: 24 }}>
                            <button className="btn btn-ghost" onClick={onClose}>Close</button>
                        </div>
                    </div>
                ) : (
                    <form onSubmit={handleSubmit}>
                        <div className="form-row mb-4">
                            <div className="form-group" style={{ marginBottom: 0 }}>
                                <label className="label">Ticket ID *</label>
                                <input name="ticket_id" value={form.ticket_id} onChange={handleChange}
                                    className="input font-mono" style={{ fontSize: 12 }} required />
                            </div>
                            <div className="form-group" style={{ marginBottom: 0 }}>
                                <label className="label">Channel</label>
                                <select name="channel" value={form.channel} onChange={handleChange} className="input select">
                                    <option value="web">Web</option>
                                    <option value="email">Email</option>
                                    <option value="phone">Phone</option>
                                    <option value="chat">Chat</option>
                                </select>
                            </div>
                        </div>

                        <div className="form-group">
                            <label className="label">Subject *</label>
                            <input name="subject" value={form.subject} onChange={handleChange}
                                className="input" placeholder="Brief description of the issue" required />
                        </div>

                        <div className="form-group">
                            <label className="label">Body *</label>
                            <textarea name="body" value={form.body} onChange={handleChange}
                                className="textarea" placeholder="Detailed description of the problem..." required />
                        </div>

                        <div className="form-group">
                            <label className="label">Customer ID <span className="text-muted">(optional)</span></label>
                            <input name="customer_id" value={form.customer_id} onChange={handleChange}
                                className="input" placeholder="CUST-123" />
                        </div>

                        {error && (
                            <div className="badge badge-red" style={{ marginBottom: 16, width: '100%', justifyContent: 'flex-start' }}>
                                <AlertTriangle size={12} /> {error}
                            </div>
                        )}

                        <div className="flex gap-3" style={{ justifyContent: 'flex-end' }}>
                            <button type="button" className="btn btn-ghost" onClick={onClose}>Cancel</button>
                            <button type="submit" className="btn btn-primary" disabled={loading}>
                                {loading ? <><Loader size={14} className="spin" /> Submitting…</> : <><Ticket size={14} /> Submit Ticket</>}
                            </button>
                        </div>
                    </form>
                )}
            </div>
        </div>
    );
}
