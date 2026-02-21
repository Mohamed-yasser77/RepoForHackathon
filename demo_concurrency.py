"""
demo_concurrency.py — Concurrent Ticket Submission Demo
========================================================
Proves that the Milestone 2 server correctly handles:
  1. High-volume concurrent requests (50+ tickets fired simultaneously)
  2. Redlock deduplication (N identical ticket_ids → only 1 enqueued)
  3. All requests safely return 202 Accepted regardless of dedup outcome

Usage:
    python demo_concurrency.py

Requires:  pip install httpx
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import List

import httpx

# ─── Configuration ────────────────────────────────────────────────────────────
# Change this when deploying to Docker/remote host.

SERVER_URL = "http://localhost:8000"

TICKETS_ENDPOINT = f"{SERVER_URL}/tickets"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"

# ─── Test Data ────────────────────────────────────────────────────────────────

SAMPLE_TICKETS = [
    {
        "subject": "Invoice overcharged me twice",
        "body": "I was billed twice this month for the same subscription. "
                "This is unacceptable, refund me ASAP!",
        "customer_id": "cust_1001",
        "channel": "email",
    },
    {
        "subject": "API returning 500 errors in production",
        "body": "Our production system is completely down. The REST API is "
                "returning 500 Internal Server Error on every request. "
                "This is critical and needs immediate attention.",
        "customer_id": "cust_1002",
        "channel": "web",
    },
    {
        "subject": "GDPR data deletion request",
        "body": "I am formally requesting the deletion of all my personal "
                "data under GDPR Article 17. Please confirm compliance "
                "within 30 days as required by law.",
        "customer_id": "cust_1003",
        "channel": "email",
    },
    {
        "subject": "Cannot log in to my account",
        "body": "The login page keeps showing a 403 Forbidden error. "
                "I have tried resetting my password but nothing works.",
        "customer_id": "cust_1004",
        "channel": "chat",
    },
    {
        "subject": "Refund for cancelled subscription",
        "body": "I cancelled my plan last week but I was still charged "
                "the full annual amount. Please process a refund.",
        "customer_id": "cust_1005",
        "channel": "phone",
    },
    {
        "subject": "Security breach on our enterprise account",
        "body": "We detected unauthorized access to our enterprise "
                "dashboard. Multiple admin accounts were compromised. "
                "This is an emergency security breach!",
        "customer_id": "cust_1006",
        "channel": "web",
    },
    {
        "subject": "Threatening legal action for data misuse",
        "body": "Your company shared my personal data with third-party "
                "advertisers without my consent. I am consulting my "
                "lawyer and will pursue legal action immediately.",
        "customer_id": "cust_1007",
        "channel": "email",
    },
    {
        "subject": "Export CSV feature is broken",
        "body": "When I try to export my reports as CSV, the downloaded "
                "file is completely empty. This has been broken for days.",
        "customer_id": "cust_1008",
        "channel": "web",
    },
    {
        "subject": "Need a copy of our enterprise SLA",
        "body": "Please provide a signed copy of our enterprise Service "
                "Level Agreement for our compliance audit.",
        "customer_id": "cust_1009",
        "channel": "email",
    },
    {
        "subject": "Webhook events not being delivered",
        "body": "Our webhook endpoint stopped receiving events since "
                "yesterday. We rely on these for order processing.",
        "customer_id": "cust_1010",
        "channel": "web",
    },
]


@dataclass
class RequestResult:
    """Outcome of a single HTTP request."""
    ticket_id: str
    status_code: int
    duplicate: bool
    elapsed_ms: float
    error: str | None = None


# ─── Helper: Send one ticket ─────────────────────────────────────────────────

async def send_ticket(
    client: httpx.AsyncClient,
    ticket_id: str,
    ticket_data: dict,
) -> RequestResult:
    """Fire a single POST /tickets and capture the result."""
    payload = {"ticket_id": ticket_id, **ticket_data}
    start = time.perf_counter()
    try:
        resp = await client.post(TICKETS_ENDPOINT, json=payload)
        elapsed = (time.perf_counter() - start) * 1000
        body = resp.json()
        return RequestResult(
            ticket_id=ticket_id,
            status_code=resp.status_code,
            duplicate=body.get("duplicate", False),
            elapsed_ms=round(elapsed, 2),
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return RequestResult(
            ticket_id=ticket_id,
            status_code=0,
            duplicate=False,
            elapsed_ms=round(elapsed, 2),
            error=str(exc),
        )


# ─── Test 1: Unique tickets fired concurrently ───────────────────────────────

async def test_concurrent_unique_tickets(client: httpx.AsyncClient) -> List[RequestResult]:
    """
    Send all 10 sample tickets simultaneously, each with a unique ticket_id.
    All should return 202 with duplicate=False.
    """
    tasks = []
    for ticket_data in SAMPLE_TICKETS:
        tid = f"tkt_{uuid.uuid4().hex[:8]}"
        tasks.append(send_ticket(client, tid, ticket_data))
    return await asyncio.gather(*tasks)


# ─── Test 2: Duplicate ticket_id storm (Redlock dedup) ────────────────────────

async def test_redlock_dedup(
    client: httpx.AsyncClient,
    copies: int = 10,
) -> List[RequestResult]:
    """
    Send `copies` identical requests with THE SAME ticket_id simultaneously.
    Exactly 1 should have duplicate=False (lock winner), the rest duplicate=True.
    """
    shared_id = f"tkt_dedup_{uuid.uuid4().hex[:6]}"
    ticket_data = SAMPLE_TICKETS[0]  # use the billing ticket
    tasks = [send_ticket(client, shared_id, ticket_data) for _ in range(copies)]
    return await asyncio.gather(*tasks)


# ─── Test 3: Mixed burst (unique + duplicates interleaved) ────────────────────

async def test_mixed_burst(client: httpx.AsyncClient) -> List[RequestResult]:
    """
    Fire 30 requests at once:
      - 10 unique tickets
      - 10 copies of ticket A (same id)
      - 10 copies of ticket B (same id)
    """
    tasks = []

    # 10 unique
    for ticket_data in SAMPLE_TICKETS:
        tid = f"tkt_mix_{uuid.uuid4().hex[:8]}"
        tasks.append(send_ticket(client, tid, ticket_data))

    # 10 duplicates of ticket A
    dup_a_id = f"tkt_dupA_{uuid.uuid4().hex[:6]}"
    for _ in range(10):
        tasks.append(send_ticket(client, dup_a_id, SAMPLE_TICKETS[1]))

    # 10 duplicates of ticket B
    dup_b_id = f"tkt_dupB_{uuid.uuid4().hex[:6]}"
    for _ in range(10):
        tasks.append(send_ticket(client, dup_b_id, SAMPLE_TICKETS[6]))

    return await asyncio.gather(*tasks)


# ─── Report Printer ──────────────────────────────────────────────────────────

def print_results(title: str, results: List[RequestResult]):
    """Pretty-print test results with stats."""
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")

    ok       = sum(1 for r in results if r.status_code == 202)
    dupes    = sum(1 for r in results if r.duplicate)
    originals = sum(1 for r in results if not r.duplicate and r.status_code == 202)
    errors   = sum(1 for r in results if r.error)
    times    = [r.elapsed_ms for r in results if r.error is None]

    for r in results:
        flag = "DUP " if r.duplicate else "NEW "
        err  = f"  ⚠ {r.error}" if r.error else ""
        print(
            f"  [{r.status_code}] {flag} {r.ticket_id:<36s} "
            f"{r.elapsed_ms:>8.1f}ms{err}"
        )

    print(f"\n  Summary:")
    print(f"    Total requests:   {len(results)}")
    print(f"    202 Accepted:     {ok}")
    print(f"    Originals (new):  {originals}")
    print(f"    Duplicates:       {dupes}")
    print(f"    Errors:           {errors}")
    if times:
        print(f"    Avg latency:      {sum(times)/len(times):.1f}ms")
        print(f"    Min / Max:        {min(times):.1f}ms / {max(times):.1f}ms")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 72)
    print("  Smart-Support Milestone 2 — Concurrency Demo")
    print(f"  Target: {SERVER_URL}")
    print("=" * 72)

    # ── Health check ──────────────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(HEALTH_ENDPOINT)
            health = resp.json()
            print(f"\n  Health: {health}")
        except Exception as exc:
            print(f"\n  ⚠ Server unreachable at {SERVER_URL}: {exc}")
            print("  Make sure the server is running: docker compose up --build")
            return

    # ── Run tests ─────────────────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1 — all unique
        print("\n▶ Test 1: 10 unique tickets fired concurrently …")
        r1 = await test_concurrent_unique_tickets(client)
        print_results("Test 1 — Concurrent Unique Tickets", r1)

        await asyncio.sleep(0.5)  # brief pause between tests

        # Test 2 — Redlock dedup (10 identical)
        print("\n▶ Test 2: 10 identical ticket_ids fired simultaneously …")
        r2 = await test_redlock_dedup(client, copies=10)
        print_results("Test 2 — Redlock Deduplication (10 copies, same ID)", r2)

        await asyncio.sleep(0.5)

        # Test 3 — Mixed burst (30 requests)
        print("\n▶ Test 3: 30-request mixed burst (unique + 2 dup groups) …")
        r3 = await test_mixed_burst(client)
        print_results("Test 3 — Mixed Burst (10 unique + 10 dupA + 10 dupB)", r3)

    # ── Final Verdict ─────────────────────────────────────────────────────
    all_results = r1 + r2 + r3
    total = len(all_results)
    all_202 = all(r.status_code == 202 for r in all_results)
    total_errs = sum(1 for r in all_results if r.error)

    print(f"\n{'=' * 72}")
    print(f"  FINAL VERDICT")
    print(f"{'=' * 72}")
    print(f"  Total requests sent:  {total}")
    print(f"  All returned 202:     {'✓ YES' if all_202 else '✗ NO'}")
    print(f"  Errors:               {total_errs}")

    # Dedup check for Test 2
    t2_originals = sum(1 for r in r2 if not r.duplicate and r.status_code == 202)
    t2_dupes     = sum(1 for r in r2 if r.duplicate)
    print(f"\n  Redlock dedup proof (Test 2):")
    print(f"    Lock winners:  {t2_originals}  (should be 1)")
    print(f"    Duplicates:    {t2_dupes}  (should be 9)")

    if all_202 and t2_originals == 1:
        print(f"\n  🎉 ALL CONCURRENCY TESTS PASSED")
    else:
        print(f"\n  ⚠  Some tests had unexpected results — review above.")

    print()


if __name__ == "__main__":
    asyncio.run(main())
