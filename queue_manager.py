"""
PriorityQueueSingleton
======================
Thread-safe singleton wrapping Python's heapq.

Heap entry: (priority, timestamp, payload_dict)
  priority  — 1=HIGH  2=MEDIUM  3=LOW  (min-heap → lower = served first)
  timestamp — UNIX float for FIFO tie-breaking within the same priority tier
"""

import heapq
import threading
from collections import Counter
from typing import Any, Dict, Optional


class PriorityQueueSingleton:
    _instance: Optional["PriorityQueueSingleton"] = None
    _cls_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "PriorityQueueSingleton":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    inst           = super().__new__(cls)
                    inst._heap     = []             # heapq list
                    inst._lock     = threading.Lock()
                    inst._pushed   = 0              # total ever pushed
                    inst._cat_cnt  = Counter()
                    inst._urg_cnt  = Counter()
                    cls._instance  = inst
        return cls._instance

    @classmethod
    def get_instance(cls) -> "PriorityQueueSingleton":
        return cls()

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, priority: int, timestamp: float, payload: Dict[str, Any]) -> int:
        """
        Push a ticket. Returns current queue depth after insertion.
        priority: 1 (HIGH) | 2 (MEDIUM) | 3 (LOW)
        """
        with self._lock:
            heapq.heappush(self._heap, (priority, timestamp, payload))
            self._pushed += 1
            self._cat_cnt[payload.get("category", "Unknown")] += 1
            self._urg_cnt[payload.get("urgency",  "Unknown")] += 1
            return len(self._heap)

    def pop(self) -> Optional[Dict[str, Any]]:
        """Remove and return the highest-priority ticket, or None if empty."""
        with self._lock:
            if not self._heap:
                return None
            _, _, payload = heapq.heappop(self._heap)
            return payload

    def peek(self) -> Optional[Dict[str, Any]]:
        """Return the next ticket without removing it."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0][2]

    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            live_urg: Dict[str, int] = {}
            for pri, _, p in self._heap:
                urg = p.get("urgency", "Unknown")
                live_urg[urg] = live_urg.get(urg, 0) + 1

            return {
                "current_queue_size":     len(self._heap),
                "total_tickets_received": self._pushed,
                "category_totals":        dict(self._cat_cnt),
                "urgency_totals":         dict(self._urg_cnt),
                "live_urgency_breakdown": live_urg,
                "next_ticket_urgency":    (
                    self._heap[0][2].get("urgency") if self._heap else None
                ),
            }