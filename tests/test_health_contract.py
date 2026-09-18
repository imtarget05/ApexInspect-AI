import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestHealthContract(unittest.TestCase):
    """Liveness/readiness split: /health/live never depends on externals,
    /health/ready reports per-dependency checks without mutating state."""

    def test_live_returns_ok_without_dependencies(self):
        from src.backend.main import health_live

        result = health_live()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["service"], "apexinspect-gateway")

    def test_ready_reports_checks_mapping(self):
        from src.backend.main import health_ready

        result = health_ready()
        self.assertIn(result["status"], ("ready", "not-ready"))
        self.assertIn("checks", result)
        self.assertIn("database", result["checks"])
        self.assertIn("model_bundle", result["checks"])


if __name__ == "__main__":
    unittest.main()
