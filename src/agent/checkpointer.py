"""JSON-safe persistent checkpointer for incident approval state.

Uses the project SQLite database (same pattern as backend/database.py):
one row per ticket/thread, JSON payload only - never serialized
executables. A fresh agent instance resumes the interrupted incident
from the persisted row.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def _db_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    url = os.environ.get("DATABASE_URL", "").strip()
    if url.startswith("sqlite") and "///" in url:
        return Path(url.split("///", 1)[1].split("?")[0])
    return Path(os.environ.get("APEX_CHECKPOINT_DB", "factory.db"))


class IncidentCheckpointer:
    """Durable incident state keyed by ticket/thread id (JSON only)."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = _db_path(db_path)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    ticket_id TEXT,
                    state_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save(self, thread_id: str, state: Dict[str, Any], ticket_id: str | None = None) -> None:
        payload = json.dumps(state, default=str, ensure_ascii=False)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO incident_checkpoints (thread_id, ticket_id, state_json)"
                " VALUES (?, ?, ?)",
                (thread_id, ticket_id or state.get("ticket_id"), payload),
            )
            conn.commit()

    def load(self, thread_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state_json FROM incident_checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row["state_json"])
        if not isinstance(data, dict):
            raise ValueError("CHECKPOINT_CORRUPT: incident state is not an object")
        return data

    def threads(self) -> List[str]:
        with self._conn() as conn:
            return [r[0] for r in conn.execute("SELECT thread_id FROM incident_checkpoints").fetchall()]

    def delete(self, thread_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM incident_checkpoints WHERE thread_id = ?", (thread_id,))
            conn.commit()
