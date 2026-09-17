"""Regression tests for the runner-independent test-database guard.

Root cause established 2026-09-17 (evidence: `.venv/bin/python -m unittest
discover -s tests -p probe.py` printed `engine_dialect = postgresql`,
`engine_db = neondb?sslmode=require`, while the same probe under
`-m pytest` printed `sqlite / factory_test.db`):

`tests/conftest.py` is a **pytest-only** hook. `src/backend/database.py` calls
`load_dotenv()`, and the repo `.env` ships a real Neon `DATABASE_URL`. So under
`python3 -m unittest discover tests/` (documented in AGENTS.md as co-equal to
pytest) the suite bound itself straight to **production Neon**, where
`tests/test_backend.py:setUp` deletes every row of `inspections` and
`mes_tickets` -> 2026-09-16 production wipe, 105s runtimes, and
`ObjectDeletedError` (ORM instance expired because a concurrent session deleted
the row that the test still held).

These tests run each runner in a subprocess with `DATABASE_URL` pointing at an
unreachable **simulated** production URL (127.0.0.1:1). Nothing here can reach
Neon: the guard must force a throwaway SQLite file no matter what the
environment/tests/conftest provide.
"""
import os
import subprocess
import sys
import textwrap
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Unreachable on purpose - simulates "the environment handed us production".
SIMULATED_PROD_URL = "postgresql://probe:probe@127.0.0.1:1/probe_never_reached"

PROBE_MODULE = textwrap.dedent(
    '''
    import os
    import unittest


    class TestProbeThrowawayBinding(unittest.TestCase):
        def test_engine_is_throwaway_sqlite(self):
            from src.backend.database import engine

            backend = engine.url.get_backend_name()
            db_file = str(engine.url)
            assert backend == "sqlite", f"bound to a remote DB: {backend} -> {db_file}"
            assert "factory_test.db" in db_file, f"not the throwaway DB: {db_file}"
            env_url = os.environ.get("DATABASE_URL", "")
            assert env_url.startswith("sqlite://"), f"env still remote: {env_url[:32]}"
    '''
)


def _child_env():
    env = dict(os.environ)
    env["DATABASE_URL"] = SIMULATED_PROD_URL
    env["PYTHONPATH"] = REPO
    # Proves the detection is runner-based, not inherited from an outer pytest.
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("APEXINSPECT_TEST_MODE", None)
    env["OPENCV_LOG_LEVEL"] = "ERROR"
    return env


def _run_probe(runner):
    with tempfile.TemporaryDirectory() as tmp:
        probe_path = os.path.join(tmp, "test_zz_throwaway_binding_probe.py")
        with open(probe_path, "w", encoding="utf-8") as f:
            f.write(PROBE_MODULE)
        if runner == "unittest":
            cmd = [sys.executable, "-m", "unittest", "discover", "-s", tmp, "-t", tmp, "-v"]
        else:
            cmd = [sys.executable, "-m", "pytest", probe_path, "-q"]
        return subprocess.run(
            cmd, cwd=REPO, env=_child_env(),
            capture_output=True, text=True, timeout=180,
        )


def _assert_passed(proc, runner):
    assert proc.returncode == 0, (
        f"{runner} bound the suite to a non-throwaway database.\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"
    )


def test_unittest_runner_uses_throwaway_database():
    """`python -m unittest discover` must never bind to a remote/production DB."""
    _assert_passed(_run_probe("unittest"), "unittest discover")


def test_pytest_runner_uses_throwaway_database():
    """pytest must stay on the throwaway DB even without tests/conftest.py."""
    _assert_passed(_run_probe("pytest"), "pytest")


def test_guard_does_not_hijack_normal_processes():
    """Production entrypoints (uvicorn/streamlit/scripts) keep their own DB URL."""
    sqlite_path = os.path.join(REPO, "factory_policy_check.db")
    script = textwrap.dedent(
        '''
        from src.backend.database import engine
        print("POLICY_ENGINE=" + str(engine.url))
        '''
    )
    with tempfile.TemporaryDirectory() as tmp:
        script_path = os.path.join(tmp, "not_a_test_script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        env = _child_env()
        env["DATABASE_URL"] = f"sqlite:///{sqlite_path}"
        try:
            proc = subprocess.run(
                [sys.executable, script_path], cwd=REPO, env=env,
                capture_output=True, text=True, timeout=120,
            )
            assert proc.returncode == 0, proc.stderr[-1500:]
            assert f"POLICY_ENGINE=sqlite:///{sqlite_path}" in proc.stdout, (
                "guard hijacked a non-test process:\n" + proc.stdout[-800:]
            )
        finally:
            for suffix in ("", "-journal", "-wal", "-shm"):
                leftover = sqlite_path + suffix
                if os.path.exists(leftover):
                    os.remove(leftover)
