"""Incident memory — learn from past interventions on the same production line.

What kind of memory this is (and is not)
----------------------------------------
ApexInspect is not a chat assistant, so "memory" here does not mean remembering
what a user said. It means the agent gets *smarter about the plant*: after
supervisors approve interventions, the next incident on the same line should not
start from zero.

Concretely, two things are remembered per line:

1. **Intervention history** — which action was taken, for which defect class,
   approved by whom. A new RCA can then say "line SMT-LINE-01 has had 3 HALT_LINE
   events for short_circuit in the last 30 days" instead of guessing.
2. **Recurrence count** — how often a defect class has appeared on a line. The
   governance rule halts a line at 3 CONSECUTIVE defects; it is a different and
   much stronger signal that the SAME defect class has recurred across days,
   which usually means the root cause was never fixed.

Design constraints
------------------
- **Advisory only.** Nothing here feeds `proposed_action`. The decision stays a
  pure function of the governance rule. If history could change the action, a
  stale or poisoned row would be able to stop a production line — exactly the
  failure mode the interrupt() gate exists to prevent.
- **SQLite, same DB as the checkpointer.** No new dependency, survives restart.
- **Never raises.** A memory read must not be able to fail an incident
  response; every helper degrades to an empty result on error.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# A defect recurring across this window on the same line means the fix did not
# hold, even if each individual cascade stayed under the consecutive threshold.
RECURRENCE_WINDOW_DAYS = 30
# Cap the context string so a decade of history cannot blow up the RCA prompt.
MAX_HISTORY_EVENTS = 5
MAX_CONTEXT_CHARS = 400


def _db_path(explicit: str | None = None) -> Path:
    """Resolve the SQLite path, mirroring checkpointer._db_path()."""
    if explicit:
        return Path(explicit)
    import os

    url = os.environ.get("DATABASE_URL", "").strip()
    if url.startswith("sqlite") and "///" in url:
        return Path(url.split("///", 1)[1].split("?")[0])
    return Path(os.environ.get("APEX_CHECKPOINT_DB", "factory.db"))


class IncidentMemory:
    """Durable per-line incident history. Advisory; never decides."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = _db_path(db_path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS incident_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        line_id TEXT NOT NULL,
                        defect_class TEXT NOT NULL,
                        action TEXT NOT NULL,
                        approval_status TEXT NOT NULL,
                        approved_by TEXT,
                        thread_id TEXT,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_mem_line ON incident_memory(line_id, created_at)"
                )
                conn.commit()
        except Exception:
            # Memory is a nice-to-have; never let init break the agent.
            pass

    def record(
        self,
        line_id: str,
        defect_class: str,
        action: str,
        approval_status: str,
        approved_by: str = "",
        thread_id: str = "",
    ) -> None:
        """Persist one resolved incident. Failures are swallowed by design."""
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO incident_memory"
                    " (line_id, defect_class, action, approval_status, approved_by, thread_id, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(line_id or "unknown"),
                        str(defect_class or "unknown"),
                        str(action or ""),
                        str(approval_status or "PENDING"),
                        str(approved_by or ""),
                        str(thread_id or ""),
                        time.time(),
                    ),
                )
                conn.commit()
        except Exception:
            pass

    def history(self, line_id: str, limit: int = MAX_HISTORY_EVENTS) -> List[Dict[str, Any]]:
        """Most recent incidents on a line, newest first. Empty on any error."""
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT line_id, defect_class, action, approval_status, approved_by, created_at"
                    " FROM incident_memory WHERE line_id = ?"
                    " ORDER BY created_at DESC LIMIT ?",
                    (str(line_id or ""), int(limit)),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def recurrence_count(self, line_id: str, defect_class: str, window_days: int = RECURRENCE_WINDOW_DAYS) -> int:
        """How often this defect class hit this line inside the window.

        Distinct from the consecutive-defect counter in `AgentState`: that one
        measures an in-flight cascade, this one measures history across shifts.
        """
        try:
            cutoff = time.time() - window_days * 86400
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM incident_memory"
                    " WHERE line_id = ? AND defect_class = ? AND created_at >= ?",
                    (str(line_id or ""), str(defect_class or ""), cutoff),
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def context_for(self, line_id: str, defect_class: str) -> str:
        """Compact Vietnamese history block for the RCA prompt.

        Returns '' when there is nothing to say, so the prompt is unchanged for
        a first-time incident rather than padded with an empty section.
        """
        events = self.history(line_id)
        if not events:
            return ""
        lines = [f"LỊCH SỬ SỰ CỐ DÂY CHUYỀN {line_id} (đã xử lý trước đó):"]
        for e in events[:MAX_HISTORY_EVENTS]:
            ts = time.strftime("%Y-%m-%d", time.localtime(e.get("created_at", 0)))
            who = f" bởi {e['approved_by']}" if e.get("approved_by") else ""
            lines.append(
                f"- {ts}: {e['defect_class']} → {e['action']} ({e['approval_status']}{who})"
            )
        count = self.recurrence_count(line_id, defect_class)
        if count >= 2:
            lines.append(
                f"LƯU Ý: lỗi '{defect_class}' đã xảy ra {count} lần trên dây chuyền này "
                f"trong {RECURRENCE_WINDOW_DAYS} ngày — nguyên nhân gốc có thể chưa được khắc phục triệt để."
            )
        return "\n".join(lines)[:MAX_CONTEXT_CHARS]

    def clear(self, line_id: str = "") -> int:
        """Delete history. Used by tests; returns rows removed."""
        try:
            with self._lock, self._connect() as conn:
                if line_id:
                    cur = conn.execute("DELETE FROM incident_memory WHERE line_id = ?", (str(line_id),))
                else:
                    cur = conn.execute("DELETE FROM incident_memory")
                conn.commit()
                return cur.rowcount
        except Exception:
            return 0
