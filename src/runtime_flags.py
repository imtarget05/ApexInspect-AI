"""Runtime mode flags shared by production code and tests.

Kept dependency-free (stdlib only) so both the backend and the agent layer can
import it without triggering heavy imports or import cycles.

Two problems this module solves, both proven by evidence in this repo:

1. Test runners must never touch the production database. ``tests/conftest.py``
   is a pytest-only hook, so ``python3 -m unittest discover tests/`` (documented
   in AGENTS.md as an equal test command) bypassed it and bound the suite to the
   real Neon instance — causing the 2026-09-16 production wipe and the
   ``ObjectDeletedError`` flake. ``src/backend/database.py`` consumes
   :func:`is_test_process` before ``load_dotenv()`` to force throwaway SQLite.

2. Test runners must never call the live LLM. ``RCAAnalysisAgent`` built its
   client from ``GROQ_API_KEY`` (loaded from ``.env``), so ordinary unit tests
   performed real HTTPS calls to ``api.groq.com`` — proven with a socket probe
   (9 external connections) and visible as 49s vs 7s suite runtimes, plus real
   429 rate-limit errors. :func:`effective_groq_key` returns an empty key under
   a test runner so the existing deterministic RCA fallback is used instead.
"""

import os
import sys

# Explicit opt-ins (documented, never set by default):
#   APEX_FORCE_TEST_MODE=1            -> pretend we are a test process
#   APEX_ALLOW_LIVE_LLM_IN_TESTS=1    -> allow real LLM calls even in a test runner
FORCE_TEST_MODE_ENV = "APEX_FORCE_TEST_MODE"
ALLOW_LIVE_LLM_ENV = "APEX_ALLOW_LIVE_LLM_IN_TESTS"
# Short alias accepted for convenience: APEX_ALLOW_LIVE_LLM=1
ALLOW_LIVE_LLM_ALIAS_ENV = "APEX_ALLOW_LIVE_LLM"


def is_test_process() -> bool:
    """True when the current interpreter is a test runner (pytest/unittest).

    Detection intentionally avoids ``sys.argv[0]`` (unreliable — shims and
    ``python -c`` rewrite it) and uses markers that cannot be produced by a
    production process:
      * ``pytest`` present in ``sys.modules`` (also ``PYTEST_CURRENT_TEST``)
      * ``__main__`` module pointing at ``unittest/__main__.py`` /
        ``unittest/main.py`` / ``pytest/__main__.py``
    """
    if os.getenv(FORCE_TEST_MODE_ENV) == "1":
        return True
    # Backward-compat alias honoured by the DB guard in database.py.
    if os.getenv("APEXINSPECT_TEST_MODE", "").strip().lower() in ("1", "true", "yes"):
        return True
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True

    if "pytest" in sys.modules:
        return True

    main_file = getattr(sys.modules.get("__main__"), "__file__", "") or ""
    normalized = main_file.replace("\\", "/")
    if normalized.endswith("unittest/__main__.py"):
        return True

    # `python -m unittest` (module form) sets the same attribute shape.
    if normalized.endswith("unittest/main.py"):
        return True

    if normalized.endswith("/pytest/__main__.py"):
        return True

    return False


def live_llm_allowed() -> bool:
    """True only when real LLM traffic is explicitly permitted.

    ``APEX_ALLOW_LIVE_LLM=1`` is accepted as a short alias for the canonical
    ``APEX_ALLOW_LIVE_LLM_IN_TESTS=1`` so both documented spellings work.
    """
    if os.getenv(ALLOW_LIVE_LLM_ENV) == "1" or os.getenv(ALLOW_LIVE_LLM_ALIAS_ENV) == "1":
        return True
    return not is_test_process()


def effective_groq_key(default: str = "") -> str:
    """Return the Groq API key, or an empty string when LLM calls are not allowed.

    An empty key makes ``RCAAnalysisAgent`` take its deterministic local
    fallback path, so unit tests stay hermetic and offline.
    """
    if not live_llm_allowed():
        return ""
    return os.getenv("GROQ_API_KEY", default)