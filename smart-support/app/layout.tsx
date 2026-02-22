import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/layout/Navbar';
import { SmoothScroll } from '@/components/animations/SmoothScroll';

export const metadata: Metadata = {
  title: 'SmartSupport — Autonomous Orchestrator',
  description: 'AI-powered customer support orchestration platform with skill-based routing, flash-flood detection, and circuit breaker resilience.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <SmoothScroll>
          <Navbar />
          {children}
        </SmoothScroll>
      </body>
    </html>
  );
}
