'use client';
import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';

interface SkillBarProps {
    label: string;
    value: number; // 0.0 - 1.0
    color?: 'green' | 'purple' | 'orange';
    delay?: number;
}

const colorClass = { green: 'skill-bar-green', purple: 'skill-bar-purple', orange: 'skill-bar-orange' };

export default function SkillBar({ label, value, color = 'green', delay = 0 }: SkillBarProps) {
    const fillRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!fillRef.current) return;
        gsap.to(fillRef.current, {
            width: `${value * 100}%`,
            duration: 1.2,
            delay,
            ease: 'power3.out',
        });
    }, [value, delay]);

    return (
        <div style={{ marginBottom: 12 }}>
            <div className="flex justify-between items-center mb-1">
                <span className="text-sm text-secondary font-semibold">{label}</span>
                <span className="font-mono text-xs text-muted">{(value * 100).toFixed(0)}%</span>
            </div>
            <div className="skill-bar-track">
                <div ref={fillRef} className={`skill-bar-fill ${colorClass[color]}`} />
            </div>
        </div>
    );
}
