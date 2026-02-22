'use client';
import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import Link from 'next/link';
import {
  Zap, ArrowRight, ShieldCheck, GitBranch, Users, AlertTriangle,
  Brain, Layers, CheckCircle
} from 'lucide-react';

const features = [
  { icon: Brain, label: 'ML-Powered Classification', desc: 'Multi-task Transformer predicts category & urgency in real-time', color: 'var(--accent-green)' },
  { icon: GitBranch, label: 'Skill-Based Routing', desc: 'Constraint optimization matches tickets to the perfect agent', color: 'var(--accent-purple)' },
  { icon: Layers, label: 'Flash-Flood Detection', desc: 'Semantic deduplication with 0.9 cosine similarity threshold', color: 'var(--accent-orange)' },
  { icon: ShieldCheck, label: 'Circuit Breaker', desc: 'Automatic failure isolation and recovery with half-open testing', color: 'var(--accent-blue)' },
  { icon: Users, label: 'Live Agent Pool', desc: 'Real-time skill vector registry with capacity management', color: 'var(--accent-purple)' },
  { icon: Zap, label: 'Async Processing', desc: '202 Accepted with ARQ background queuing — never blocks', color: 'var(--accent-green)' },
];

const stats = [
  { value: '3.0', label: 'App Version', suffix: '' },
  { value: '0.9', label: 'Similarity Threshold', suffix: '' },
  { value: '5', label: 'Second Redlock Window', suffix: 's' },
  { value: '3', label: 'ML Categories', suffix: '' },
];

export default function HomePage() {
  const heroRef = useRef<HTMLDivElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const descRef = useRef<HTMLParagraphElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);
  const pillsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!heroRef.current) return;

    const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });
    tl.fromTo('.hero-eyebrow', { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.6 })
      .fromTo(titleRef.current, { opacity: 0, y: 40, scale: 0.95 }, { opacity: 1, y: 0, scale: 1, duration: 0.9 }, '-=0.3')
      .fromTo(descRef.current, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.6 }, '-=0.5')
      .fromTo(ctaRef.current, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.6 }, '-=0.4')
      .fromTo('.stat-pill', { opacity: 0, y: 20, scale: 0.9 }, { opacity: 1, y: 0, scale: 1, stagger: 0.07, duration: 0.5 }, '-=0.2')
      .fromTo('.feature-card-item', { opacity: 0, y: 50 }, { opacity: 1, y: 0, stagger: 0.08, duration: 0.6 }, '-=0.2');
  }, []);

  return (
    <main>
      {/* ── HERO ── */}
      <section className="hero" ref={heroRef}>
        <div className="hero-orb orb-1" />
        <div className="hero-orb orb-2" />
        <div className="hero-orb orb-3" />

        <div className="hero-eyebrow">
          <Zap size={12} />
          Autonomous Support Orchestrator · v3.0.0
        </div>

        <h1 ref={titleRef} className="hero-title">
          Support That<br />
          <span className="highlight">Thinks Faster</span><br />
          Than Humans
        </h1>

        <p ref={descRef} className="hero-description">
          Smart-Support combines ML classification, semantic deduplication, and constraint optimization to automatically route tickets to the best available agent — before a human even reads it.
        </p>

        <div ref={ctaRef} className="hero-cta">
          <Link href="/dashboard" className="btn btn-primary btn-lg">
            Open Dashboard <ArrowRight size={18} />
          </Link>
          <Link href="/tickets" className="btn btn-ghost btn-lg">
            Submit Ticket
          </Link>
        </div>

        {/* Live stats pills */}
        <div className="feature-pills" style={{ marginTop: 48 }}>
          {stats.map(({ value, label, suffix }) => (
            <div key={label} className="feature-pill stat-pill">
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-green)', fontSize: 18 }}>
                {value}{suffix}
              </span>
              <span className="text-sm text-secondary">{label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── FEATURES ── */}
      <section style={{ maxWidth: 1200, margin: '0 auto', padding: '0 40px 100px' }}>
        <div style={{ textAlign: 'center', marginBottom: 48 }}>
          <div className="badge badge-purple" style={{ marginBottom: 14, display: 'inline-flex' }}>
            <CheckCircle size={10} /> Platform Capabilities
          </div>
          <h2 style={{ fontSize: 'clamp(28px,3vw,40px)', fontWeight: 700, letterSpacing: '-0.03em', color: 'var(--text-primary)', marginBottom: 12 }}>
            Everything You Need to Scale Support
          </h2>
          <p style={{ color: 'var(--text-secondary)', maxWidth: 520, margin: '0 auto', lineHeight: 1.6 }}>
            Three milestones of engineering combined into a single resilient pipeline.
          </p>
        </div>

        <div className="grid-3">
          {features.map(({ icon: Icon, label, desc, color }) => (
            <div key={label} className="card feature-card-item" style={{ cursor: 'default' }}>
              <div style={{
                width: 44, height: 44, borderRadius: 10, marginBottom: 16,
                background: `${color}18`, border: `1px solid ${color}30`,
                display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}>
                <Icon size={20} color={color} />
              </div>
              <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8, color: 'var(--text-primary)' }}>{label}</h3>
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{desc}</p>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
