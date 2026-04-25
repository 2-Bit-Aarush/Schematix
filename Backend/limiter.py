"""
limiter.py
----------
Persistent LLM call counter with a simple two-function public API.

Public API:
  can_call_llm()     -> bool   — check if a call is permitted (does NOT increment)
  increment_usage()  -> None   — record that one LLM call was made

Design:
  - Zero dependencies beyond stdlib
  - Survives server restarts (persists to data/llm_usage.json)
  - Single-process safe (FastAPI's asyncio loop is single-threaded by default)
  - Limit is read from MAX_LLM_CALLS env variable (default 1000)
  - Never raises exceptions — returns False / logs on any error

Usage pattern in callers:
    if limiter.can_call_llm():
        limiter.increment_usage()
        result = llm.enhance_query(q)
    else:
        result = q   # graceful fallback
        warning = "LLM limit reached, using basic search"

Keeping check and increment separate (rather than a combined check_and_increment)
makes caller intent explicit and avoids accidental double-counting.

Storage schema — data/llm_usage.json:
  { "total_calls": int, "limit": int }
"""

import json
import os

USAGE_PATH    = "data/llm_usage.json"
_DEFAULT_LIMIT = 1000

# Read once at import time. Changing MAX_LLM_CALLS requires a server restart,
# which is the standard operational contract for environment-driven config.
_LIMIT = int(os.getenv("MAX_LLM_CALLS", str(_DEFAULT_LIMIT)))


class UsageLimiter:
    """
    Tracks cumulative LLM API calls across the full server lifetime.

    Typical lifecycle:
        limiter = UsageLimiter()            # load from disk

        if limiter.can_call_llm():
            limiter.increment_usage()       # persist immediately
            answer = llm.generate_answer(q, ctx)
        else:
            answer = FALLBACK_MESSAGE
    """

    def __init__(self):
        self.limit: int = _LIMIT
        self.total_calls: int = 0
        self._load()
        print(
            f"[Limiter] Limit: {self.limit} LLM calls. "
            f"Used so far: {self.total_calls}. "
            f"Remaining: {max(0, self.limit - self.total_calls)}."
        )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def can_call_llm(self) -> bool:
        """
        Return True if at least one more LLM call is permitted.

        Does NOT increment the counter — call increment_usage() separately
        immediately before the actual LLM call to keep accounting honest.
        """
        return self.total_calls < self.limit

    def increment_usage(self) -> None:
        """
        Record that one LLM API call is about to be made (or was just made).
        Persists to disk immediately so a crash doesn't lose the count.

        Call this BEFORE the LLM call so that even failed/errored calls are
        counted — this prevents budget bleed from retry storms.
        """
        self.total_calls += 1
        self._save()

    # ------------------------------------------------------------------
    # Status / admin
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """
        Return a serialisable status dict suitable for inclusion in API responses.
        Frontend can display this in a usage banner.
        """
        return {
            "llm_calls_used":      self.total_calls,
            "llm_calls_limit":     self.limit,
            "llm_calls_remaining": max(0, self.limit - self.total_calls),
            "llm_available":       self.can_call_llm(),
        }

    def reset(self) -> None:
        """Reset the counter to zero. Expose via an admin endpoint if desired."""
        self.total_calls = 0
        self._save()
        print("[Limiter] Usage counter reset to 0.")

    # ------------------------------------------------------------------
    # Internal persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load persisted counter from disk. Initialises to 0 if file is missing/corrupt."""
        if not os.path.exists(USAGE_PATH):
            self.total_calls = 0
            return
        try:
            with open(USAGE_PATH, "r") as f:
                data = json.load(f)
            self.total_calls = int(data.get("total_calls", 0))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            print("[Limiter] WARNING: usage file is corrupted — resetting to 0.")
            self.total_calls = 0

    def _save(self) -> None:
        """Flush current counter to disk."""
        os.makedirs("data", exist_ok=True)
        with open(USAGE_PATH, "w") as f:
            json.dump(
                {"total_calls": self.total_calls, "limit": self.limit},
                f,
                indent=2,
            )
