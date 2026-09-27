"""Tests for advisory incident memory.

The invariant that matters
--------------------------
`IncidentMemory` is allowed to change the RCA *narrative* and nothing else. If
history could change `proposed_action`, a stale or poisoned row would be able to
stop a production line - exactly the failure the interrupt() gate exists to
prevent. These tests assert that invariant directly, not just that rows store.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.memory import (
    MAX_CONTEXT_CHARS,
    IncidentMemory,
    RECURRENCE_WINDOW_DAYS,
)
from src.agent.graph import QualityIncidentAgent

LINE = "SMT-LINE-01"


class MemoryTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "mem.db")
        self.mem = IncidentMemory(db_path=self.db)


class TestRecordAndRecall(MemoryTestBase):
    def test_history_round_trips(self):
        self.mem.record(LINE, "short_circuit", "HALT_LINE", "APPROVED", "sup_a")
        events = self.mem.history(LINE)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["defect_class"], "short_circuit")
        self.assertEqual(events[0]["action"], "HALT_LINE")
        self.assertEqual(events[0]["approved_by"], "sup_a")

    def test_history_is_scoped_per_line(self):
        self.mem.record(LINE, "short_circuit", "HALT_LINE", "APPROVED")
        self.mem.record("SMT-LINE-02", "missing_hole", "ROUTE_REWORK", "APPROVED")
        self.assertEqual(len(self.mem.history(LINE)), 1)
        self.assertEqual(len(self.mem.history("SMT-LINE-02")), 1)

    def test_survives_a_new_instance(self):
        self.mem.record(LINE, "short_circuit", "HALT_LINE", "APPROVED")
        self.assertEqual(len(IncidentMemory(db_path=self.db).history(LINE)), 1)

    def test_newest_first(self):
        for i in range(3):
            self.mem.record(LINE, f"defect_{i}", "HALT_LINE", "APPROVED")
        self.assertEqual(self.mem.history(LINE)[0]["defect_class"], "defect_2")

    def test_history_is_capped(self):
        for i in range(12):
            self.mem.record(LINE, f"d{i}", "HALT_LINE", "APPROVED")
        self.assertEqual(len(self.mem.history(LINE)), 5)

    def test_unknown_line_returns_empty(self):
        self.assertEqual(self.mem.history("nope"), [])


class TestRecurrence(MemoryTestBase):
    def test_counts_only_matching_line_and_defect(self):
        self.mem.record(LINE, "short_circuit", "HALT_LINE", "APPROVED")
        self.mem.record(LINE, "missing_hole", "ROUTE_REWORK", "APPROVED")
        self.assertEqual(self.mem.recurrence_count(LINE, "short_circuit"), 1)
        self.assertEqual(self.mem.recurrence_count(LINE, "missing_hole"), 1)
        self.assertEqual(self.mem.recurrence_count("OTHER", "short_circuit"), 0)

    def test_window_excludes_old_events(self):
        import sqlite3
        import time

        self.mem.record(LINE, "short_circuit", "HALT_LINE", "APPROVED")
        stale = time.time() - (RECURRENCE_WINDOW_DAYS + 5) * 86400
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE incident_memory SET created_at = ?", (stale,))
        self.assertEqual(self.mem.recurrence_count(LINE, "short_circuit"), 0)


class TestContextFor(MemoryTestBase):
    def test_empty_for_a_first_time_incident(self):
        self.assertEqual(self.mem.context_for(LINE, "short_circuit"), "")

    def test_mentions_line_and_defect(self):
        self.mem.record(LINE, "short_circuit", "HALT_LINE", "APPROVED", "sup_a")
        ctx = self.mem.context_for(LINE, "short_circuit")
        self.assertIn(LINE, ctx)
        self.assertIn("short_circuit", ctx)
        self.assertIn("HALT_LINE", ctx)

    def test_no_recommendation_without_repetition(self):
        """A single past event must not be phrased as a pattern."""
        self.mem.record(LINE, "short_circuit", "HALT_LINE", "APPROVED")
        self.assertNotIn("LƯU Ý", self.mem.context_for(LINE, "short_circuit"))

    def test_repetition_is_flagged(self):
        self.mem.record(LINE, "short_circuit", "HALT_LINE", "APPROVED")
        self.mem.record(LINE, "short_circuit", "HALT_LINE", "APPROVED")
        self.assertIn("LƯU Ý", self.mem.context_for(LINE, "short_circuit"))

    def test_context_is_length_capped(self):
        for i in range(20):
            self.mem.record(LINE, f"a_very_long_defect_class_name_{i}", "HALT_LINE", "APPROVED")
        self.assertLessEqual(len(self.mem.context_for(LINE, "short_circuit")), MAX_CONTEXT_CHARS)


class _BrokenMemory:
    """Stand-in for a memory backend that fails on every call."""

    def context_for(self, line_id, defect_class):
        raise RuntimeError("memory backend down")

    def record(self, *a, **k):
        raise RuntimeError("memory backend down")


class TestMemoryNeverBreaksTheAgent(MemoryTestBase):
    """A memory outage must degrade to 'no history', not fail an incident."""

    def test_unwritable_db_degrades_quietly(self):
        broken = IncidentMemory(db_path="/proc/definitely-not-writable/x.db")
        broken.record(LINE, "d", "HALT_LINE", "APPROVED")
        self.assertEqual(broken.history(LINE), [])
        self.assertEqual(broken.context_for(LINE, "d"), "")

    def test_node_returns_empty_context_on_failure(self):
        agent = QualityIncidentAgent()
        agent.memory = _BrokenMemory()
        out = agent._node_load_history({"line_id": LINE, "defect_class": "short_circuit"})
        self.assertEqual(out["history_context"], "")


class TestMemoryDoesNotChangeDecisions(unittest.TestCase):
    """THE invariant: history is advisory, never a decision input."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "agent.db")
        self.agent = QualityIncidentAgent()
        self.agent.memory = IncidentMemory(db_path=self.db)

    def _decide(self, count):
        return self.agent.run(defect_class="short_circuit", consecutive_count=count, line_id=LINE)

    def test_decision_identical_with_and_without_history(self):
        cold = self._decide(3)["proposed_action"]
        # Same agent, now with a heavy history on this very line.
        for _ in range(5):
            self.agent.memory.record(LINE, "short_circuit", "ROUTE_REWORK", "APPROVED")
        warm = self._decide(3)["proposed_action"]
        self.assertEqual(cold, warm, "memory must not alter the governance decision")

    def test_rejected_history_does_not_soften_the_halt(self):
        """Even a wall of REJECTED rows must not downgrade HALT_LINE."""
        for _ in range(10):
            self.agent.memory.record(LINE, "short_circuit", "ROUTE_REWORK", "REJECTED")
        self.assertEqual(self._decide(3)["proposed_action"], "HALT_LINE")

    def test_history_reaches_the_rca_prompt_only(self):
        self.agent.memory.record(LINE, "short_circuit", "HALT_LINE", "APPROVED")
        out = self._decide(2)
        self.assertIn("LỊCH SỬ", out["history_context"])
        # Deterministic mode ignores history in the RCA text, keeping the eval
        # gate's content/hallucination scoring reproducible.
        self.assertNotIn("LỊCH SỬ", out["rca_analysis"])

    def test_cold_start_has_no_history_context(self):
        self.assertEqual(self._decide(1).get("history_context", ""), "")


if __name__ == "__main__":
    unittest.main()
