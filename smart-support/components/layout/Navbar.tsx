'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Zap, LayoutDashboard, Ticket, AlertTriangle, Users } from 'lucide-react';
import { clsx } from 'clsx';

const navItems = [
    { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { href: '/tickets', label: 'Tickets', icon: Ticket },
    { href: '/incidents', label: 'Incidents', icon: AlertTriangle },
    { href: '/agents', label: 'Agents', icon: Users },
];

export default function Navbar() {
    const pathname = usePathname();

    return (
        <header className="navbar">
            <Link href="/" className="navbar-logo" style={{ textDecoration: 'none', color: 'var(--text-primary)' }}>
                <div className="navbar-logo-icon">
                    <Zap size={16} color="#080A10" fill="#080A10" />
                </div>
                <span>Smart<span style={{ color: 'var(--accent-green)' }}>Support</span></span>
            </Link>

            <nav>
                <ul className="navbar-nav">
                    {navItems.map(({ href, label, icon: Icon }) => (
                        <li key={href}>
                            <Link
                                href={href}
                                className={clsx('navbar-link', pathname.startsWith(href) && 'active')}
                            >
                                <Icon size={14} />
                                {label}
                            </Link>
                        </li>
                    ))}
                </ul>
            </nav>

            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="badge badge-green">
                    <span className="dot dot-green" />
                    v3.0.0
                </span>
            </div>
        </header>
    );
}
