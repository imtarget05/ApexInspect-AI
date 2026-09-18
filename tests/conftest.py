"""Global pytest configuration.

CRITICAL: the test suite must never touch the production Neon database.
`tests/test_backend.py` truncates the `inspections` / `mes_tickets` tables in
`setUp`, and `.env` ships a real `DATABASE_URL` for Neon — which silently turned
every `pytest` run into a production wipe. This session-wide override points the
backend at a throwaway local SQLite file before any `src.*` module is imported
(`load_dotenv()` does not override pre-existing environment variables).

Vision tests (test_ingest.py, test_prepare_dataset.py, test_vision.py,
test_quality.py, test_stream.py, test_sample_gallery.py, and any test that
imports src/vision or src/ui.app) require OpenCV (cv2). On machines where cv2
is not installed these tests are skipped automatically so that `pytest -q`
does not choke the whole suite at collection time. On a machine with cv2
installed, install it with `pip install opencv-python` (or the project's
declared extra) and re-run.
"""
import os
import importlib.util
import pytest

os.environ["DATABASE_URL"] = "sqlite:///./factory_test.db"
# LLM guard: the suite must never open a socket to api.groq.com.
# `RCAAnalysisAgent.run()` treats this exact placeholder as "not configured" and
# returns its deterministic template, so no network call happens. `load_dotenv()`
# never overrides a pre-existing variable, so setting it here (before any `src.*`
# import) is sufficient. Without it a full run made 9 live HTTPS calls to Groq and
# wall-clock time swung between 7.4s and 49.6s depending on API latency/quota.
os.environ["GROQ_API_KEY"] = "gsk_your_groq_api_key_here"

VISION_AVAILABLE = importlib.util.find_spec("cv2") is not None
"""True when OpenCV is installed.

Tests that import a cv2-dependent module are skipped when this is False. The
list below is derived from `grep -rn 'import cv2' src/ tests/` at plan 시간을
(2026-09-18) and is maintained by the repo's CI/docs; if a new cv2-dependent
test is added, add its file name here (or add a `pytest.mark.vision` marker and
this hook will pick any item carrying that marker).
"""
_CV2_DEPENDENT_TEST_FILES = frozenset(
    {
        "test_ingest.py",
        "test_prepare_dataset.py",
        "test_vision.py",
        "test_quality.py",
        "test_stream.py",
        "test_sample_gallery.py",
        "test_patching.py",
    }
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "vision: mark test as requiring OpenCV (cv2). Skipped when not installed.",
    )


def pytest_ignore_collect(collection_path, config):
    """Prevent collecting files that import cv2 at the module level when cv2 is missing.

    This runs BEFORE collection, so cv2-dependent test files are never even
    imported — avoiding the ModuleNotFoundError at collection time (not at
    test-execution time).
    """
    if not VISION_AVAILABLE:
        # collection_path can be a str or a pathlib.Path depending on pytest version.
        name = getattr(collection_path, "name", None) or str(collection_path).split(os.sep)[-1]
        if name in _CV2_DEPENDENT_TEST_FILES:
            return True
    return False


def pytest_collection_modifyitems(config, items):
    if not VISION_AVAILABLE:
        skip_vision = pytest.mark.skip(reason="opencv (cv2) not installed")
        for item in items:
            if "vision" in item.keywords:
                item.add_marker(skip_vision)

