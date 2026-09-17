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
# LLM guard: the suite must never open a socket to api.groq.com.
# `RCAAnalysisAgent.run()` treats this exact placeholder as "not configured" and
# returns its deterministic template, so no network call happens. `load_dotenv()`
# never overrides a pre-existing variable, so setting it here (before any `src.*`
# import) is sufficient. Without it a full run made 9 live HTTPS calls to Groq and
# wall-clock time swung between 7.4s and 49.6s depending on API latency/quota.
os.environ["GROQ_API_KEY"] = "gsk_your_groq_api_key_here"
