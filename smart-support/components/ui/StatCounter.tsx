'use client';
import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';

interface StatCounterProps {
    value: number | string;
    label: string;
    color?: 'green' | 'purple' | 'orange';
    suffix?: string;
    prefix?: string;
    decimals?: number;
}

export default function StatCounter({ value, label, color = 'green', suffix = '', prefix = '', decimals = 0 }: StatCounterProps) {
    const numRef = useRef<HTMLSpanElement>(null);

    useEffect(() => {
        if (typeof value !== 'number' || !numRef.current) return;
        const obj = { val: 0 };
        gsap.to(obj, {
            val: value,
            duration: 1.8,
            ease: 'power3.out',
            onUpdate: () => {
                if (numRef.current) {
                    numRef.current.textContent = obj.val.toFixed(decimals);
                }
            },
        });
    }, [value, decimals]);

    return (
        <div>
            <div className={`stat-number stat-number-${color}`}>
                {prefix}
                <span ref={numRef}>{typeof value === 'number' ? '0' : value}</span>
                {suffix}
            </div>
            <p className="text-sm text-secondary" style={{ marginTop: 6, fontWeight: 500 }}>{label}</p>
        </div>
    );
}
