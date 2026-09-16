"""Regression guard: the test suite must not point at the production database.

`tests/test_backend.py:setUp` deletes every row of `inspections` and
`mes_tickets`. When `.env` provides a real Neon `DATABASE_URL`, that turned a
green `pytest` run into a production wipe (55 inspection rows lost on
2026-09-16). These tests fail loudly if the isolation ever regresses.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_engine_is_local_sqlite_not_remote():
    from src.backend.database import engine

    assert engine.url.get_backend_name() == "sqlite", (
        "Tests must never run against a remote database - production rows "
        f"would be deleted. Got: {engine.url.get_backend_name()}"
    )
    assert "factory_test.db" in str(engine.url), (
        "Tests must use the dedicated throwaway DB (tests/conftest.py), "
        f"got: {engine.url}"
    )


def test_conftest_overrides_dotenv_database_url():
    """`load_dotenv()` must not silently re-inject the Neon URL into tests."""
    assert os.environ["DATABASE_URL"].startswith("sqlite://"), (
        f"Expected the test override to survive .env loading, got {os.environ['DATABASE_URL'][:20]}..."
    )
