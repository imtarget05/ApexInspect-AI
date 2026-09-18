import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.database import Base
from src.backend import models  # noqa: F401  (registers tables)


class TestWorkflowTables(unittest.TestCase):
    """Shared durable-ops contract: idempotency, jobs, outbox, DLQ,
    audit events, and model registry tables must exist on Base metadata."""

    def test_shared_contract_tables_registered(self):
        names = set(Base.metadata.tables.keys())
        for expected in (
            "idempotency_keys",
            "jobs",
            "outbox_events",
            "dead_letters",
            "audit_events",
            "model_registry",
        ):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
