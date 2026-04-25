"""
history.py
----------
Persistent search history — stores every query made through /search and /ask.

Public API:
  add(query, user, top_result)  -> None   — append one entry, flush to disk
  get(user)                     -> List[dict]   — fetch entries, newest first

Storage: data/search_history.json
  A JSON array where each entry is:
  {
    "query":      str,           # the raw query (before enhancement)
    "timestamp":  str,           # ISO-8601 UTC, e.g. "2024-11-01T14:32:05Z"
    "user":       str,           # "guest" by default; extend for auth later
    "top_result": str            # first chunk text, or "" if no results
  }

Design:
  - File is created on first write — safe to delete between runs
  - Handles missing / empty / corrupted file gracefully (starts fresh)
  - Returns entries in reverse chronological order (newest first)
  - No external dependencies beyond stdlib
  - Not thread-safe for multi-process deployments — fine for single-process FastAPI
"""

import json
import os
from datetime import datetime, timezone
from typing import List

HISTORY_PATH = "data/search_history.json"


class SearchHistory:
    """
    In-memory search history backed by a JSON file.

    The full history is loaded into memory at startup and kept in sync with
    disk on every write. For very large deployments (millions of entries) you
    would switch to a database, but for a RAG API serving hundreds to thousands
    of queries this is fast and operationally simple.
    """

    def __init__(self):
        self._entries: List[dict] = []
        self._load()
        print(f"[History] Loaded {len(self._entries)} historical entries.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, query: str, user: str = "guest", top_result: str = "") -> None:
        """
        Append a new search entry and immediately persist to disk.

        Args:
            query:      The user's raw search query (pre-enhancement).
            user:       User identifier — default "guest" until auth is added.
            top_result: Text of the top-ranked chunk, or "" if no results.
                        Useful for building a "what people found" display.
        """
        entry = {
            "query":      query.strip(),
            "timestamp":  _utc_now(),
            "user":       user or "guest",
            "top_result": top_result or "",
        }
        self._entries.append(entry)
        self._save()

    def get(self, user: str = "") -> List[dict]:
        """
        Return search history entries in reverse chronological order (newest first).

        Args:
            user: If non-empty, filter to entries for that user only.
                  If empty (default), return all entries.

        Returns:
            List of entry dicts, newest first.
        """
        if user:
            filtered = [e for e in self._entries if e.get("user") == user]
        else:
            filtered = list(self._entries)

        # Reverse so the caller always sees the most recent entry at index 0
        return list(reversed(filtered))

    def clear(self, user: str = "") -> int:
        """
        Delete history entries. Optionally scoped to a specific user.
        Returns the number of entries removed.
        Useful for a future admin / privacy endpoint.
        """
        before = len(self._entries)
        if user:
            self._entries = [e for e in self._entries if e.get("user") != user]
        else:
            self._entries = []
        removed = before - len(self._entries)
        if removed:
            self._save()
        return removed

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Load entries from disk.
        Silently starts with an empty list if the file is missing or malformed.
        """
        if not os.path.exists(HISTORY_PATH):
            self._entries = []
            return
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Validate: must be a list; bad entries are skipped rather than crashing
            if isinstance(data, list):
                self._entries = [e for e in data if isinstance(e, dict)]
            else:
                print("[History] WARNING: history file has unexpected format — starting fresh.")
                self._entries = []
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[History] WARNING: could not load history ({exc}) — starting fresh.")
            self._entries = []

    def _save(self) -> None:
        """Persist the full entry list to disk atomically-ish (write + rename)."""
        os.makedirs("data", exist_ok=True)
        tmp_path = HISTORY_PATH + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, ensure_ascii=False)
            # Atomic rename so a crash mid-write doesn't corrupt the file
            os.replace(tmp_path, HISTORY_PATH)
        except OSError as exc:
            print(f"[History] ERROR: could not save history ({exc}).")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return current UTC time as an ISO-8601 string with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
