"""Global pytest configuration.

CRITICAL: the test suite must never touch the production Neon database.
`tests/test_backend.py` truncates the `inspections` / `mes_tickets` tables in
`setUp`, and `.env` ships a real `DATABASE_URL` for Neon — which silently turned
every `pytest` run into a production wipe. This session-wide override points the
backend at a throwaway local SQLite file before any `src.*` module is imported
(`load_dotenv()` does not override pre-existing environment variables).
"""
import os

os.environ["DATABASE_URL"] = "sqlite:///./factory_test.db"
