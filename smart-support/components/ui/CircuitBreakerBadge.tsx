'use client';
import { ShieldCheck, ShieldAlert, ShieldQuestion } from 'lucide-react';

type CBState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

interface CircuitBreakerBadgeProps {
    state: CBState;
}

const stateMap = {
    CLOSED: {
        cls: 'circuit-closed',
        icon: ShieldCheck,
        label: 'CLOSED — Nominal',
        dot: 'dot-green',
    },
    OPEN: {
        cls: 'circuit-open',
        icon: ShieldAlert,
        label: 'OPEN — Tripped',
        dot: 'dot-red',
    },
    HALF_OPEN: {
        cls: 'circuit-half',
        icon: ShieldQuestion,
        label: 'HALF-OPEN — Testing',
        dot: 'dot-orange',
    },
};

export default function CircuitBreakerBadge({ state }: CircuitBreakerBadgeProps) {
    const { cls, icon: Icon, label, dot } = stateMap[state] ?? stateMap.CLOSED;
    return (
        <div className={`circuit-breaker ${cls}`}>
            <span className={`dot ${dot}`} />
            <Icon size={14} />
            CIRCUIT BREAKER: {label}
        </div>
    );
}
