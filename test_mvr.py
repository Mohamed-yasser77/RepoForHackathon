"""
test_mvr.py — Offline smoke test (no server needed).
Run AFTER download_and_train.py (or it will train on seed data only).

Usage:  python test_mvr.py
"""

import re
import sys
import time

from classifier import EnsembleIRClassifier
from queue_manager import PriorityQueueSingleton

# ── Test cases: (text, expected_category, expected_urgency) ──────────────────

TESTS = [
    ("Invoice shows wrong amount I was charged twice this month",                   "Billing",   "LOW"),
    ("My payment method was declined but money was still deducted ASAP please fix", "Billing",   "HIGH"),
    ("The API is returning 500 errors our production is completely down",           "Technical", "HIGH"),
    ("Browser extension crashes on Chrome latest update",                           "Technical", "LOW"),
    ("I am requesting data deletion under GDPR Article 17 formal complaint",        "Legal",     "LOW"),
    ("Threatening legal action if not resolved within 48 hours",                    "Legal",     "HIGH"),
    ("How much does the pro plan cost please send pricing",                         "Billing",   "LOW"),
    ("Security breach detected on our account please help immediately",             "Technical", "HIGH"),
    ("Your terms of service changed without proper notice I need legal review",     "Legal",     "LOW"),
    ("SSO login throwing SAML assertion error cannot access urgent",                "Technical", "HIGH"),
]

_HIGH_RE = re.compile(
    r"\b(asap|urgent|urgently|immediately|critical|emergency|broken|outage|"
    r"not working|cannot access|data loss|security breach|compromised|"
    r"lawsuit|legal action|refund now|escalate|production|sev[- ]?1|"
    r"threatening|down|unresponsive|unreachable)\b", re.IGNORECASE,
)
_MED_RE  = re.compile(
    r"\b(slow|delay|error|fail|issue|problem|bug|incorrect|wrong|"
    r"disappointed|frustrated|waiting|pending|days|week)\b", re.IGNORECASE,
)

def urgency(text):
    h = len(_HIGH_RE.findall(text))
    m = len(_MED_RE.findall(text))
    if h >= 1: return "HIGH"
    if m >= 1: return "MEDIUM"
    return "LOW"


def main():
    print("=" * 65)
    print("  Smart-Support MVR — Smoke Test")
    print("=" * 65)

    clf = EnsembleIRClassifier()
    clf.load_or_train()      # loads cached model or trains fresh

    q = PriorityQueueSingleton.get_instance()

    print("\n── Classification + Queue ─────────────────────────────────────")
    passed = failed = 0
    for text, exp_cat, exp_urg in TESTS:
        cat, conf, votes = clf.predict(text)
        urg = urgency(text)

        cat_ok = cat == exp_cat
        urg_ok = urg == exp_urg
        ok     = cat_ok and urg_ok
        mark   = "✓" if ok else "✗"
        if ok: passed += 1
        else:  failed += 1

        pri = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}[urg]
        q.push(pri, time.time(), {"category": cat, "urgency": urg, "text": text})

        print(f"  {mark}  [{cat:10s}|{urg:6s}] conf={conf:.3f}")
        if not ok:
            if not cat_ok: print(f"      ↳ category: got '{cat}'  expected '{exp_cat}'")
            if not urg_ok: print(f"      ↳ urgency:  got '{urg}'  expected '{exp_urg}'")
        print(f"      Text: {text[:72]}")
        print(f"      Votes: {votes}")

    print(f"\n  Result: {passed}/{passed+failed} passed")

    print("\n── Queue Stats ────────────────────────────────────────────────")
    for k, v in q.stats().items():
        print(f"  {k}: {v}")

    print("\n── Dequeue (should start with HIGH) ──────────────────────────")
    for _ in range(min(5, q.size())):
        item = q.pop()
        print(f"  [{item['urgency']:6s}] {item['text'][:65]}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())