"""Guards so a test can never write into the real logs/llm-telemetry.jsonl."""
import os
from pathlib import Path

import pytest

import server

REAL_LOG = Path(__file__).resolve().parents[1] / "logs" / "llm-telemetry.jsonl"


@pytest.fixture(autouse=True)
def protect_real_telemetry_log():
    before = REAL_LOG.read_bytes() if REAL_LOG.exists() else None
    yield
    after = REAL_LOG.read_bytes() if REAL_LOG.exists() else None
    assert before == after, f"a test wrote to the real {REAL_LOG} — point LLM_LOG_PATH at tmp_path"
    assert os.environ.get("LLM_LOG_PATH") is None or "pytest" in os.environ.get("LLM_LOG_PATH", "") \
        or Path(os.environ["LLM_LOG_PATH"]).parent != REAL_LOG.parent
