"""Process-wide daily usage budgets for billable cloud calls (cost guards).

Keys are free-form ("vision_pages", "prefix_tokens"); a budget resets at UTC midnight.
Consumption is all-or-nothing per request so a file is either fully admitted or refused
up front, never half-parsed for budget reasons.

Kept in memory on purpose: it is a runaway guard (a loop, a huge upload, a mis-set
cap), not billing. A restart resets the day's counters — documented in the README;
a persisted usage table would be the follow-up if exact accounting is ever needed.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timezone
from typing import Callable, Dict


class BudgetExceeded(RuntimeError):
    """A daily budget cannot admit the requested amount."""


class DailyBudget:
    def __init__(self, today: Callable[[], date] = lambda: datetime.now(timezone.utc).date()):
        self._today = today
        self._day = today()
        self._used: Dict[str, int] = {}
        self._lock = threading.Lock()

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day = day
            self._used = {}

    def try_consume(self, key: str, amount: int, limit: int) -> bool:
        """Record ``amount`` against ``key`` if it fits ``limit`` (0 = unlimited)."""
        with self._lock:
            self._roll()
            used = self._used.get(key, 0)
            if limit and used + amount > limit:
                return False
            self._used[key] = used + amount
            return True

    def consume_or_raise(self, key: str, amount: int, limit: int, what: str) -> None:
        if not self.try_consume(key, amount, limit):
            raise BudgetExceeded(
                f"daily budget for {what} exhausted: {self.used(key)}/{limit} used today (UTC), "
                f"{amount} more requested — raise the cap or retry tomorrow")

    def used(self, key: str) -> int:
        with self._lock:
            self._roll()
            return self._used.get(key, 0)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            self._roll()
            return dict(self._used)


BUDGET = DailyBudget()
