"""VCR replay test for Groq RCA (beyond-mock).

Uses a recorded HTTP cassette instead of mocks, so the test exercises the
real ``CloudGroqProvider`` HTTP path on replay without touching the network.

Contract:
- cassette dir: ``tests/cassettes/``
- ``record_mode="once"``: replay if the cassette exists, record only once
  when it is missing (requires a real key + explicit opt-in).
- ``filter_headers`` strips ``Authorization`` so no secret is recorded.
- Graceful skip when ``vcrpy`` is not installed.
- Never breaks the anti-mock guard in ``tests/test_llm_guard.py``: this
  module makes no live Groq call by default (skips when the cassette is
  absent and no real key is present).
"""

import os

import pytest

try:
    import vcr
except ImportError:  # graceful skip if vcrpy missing
    vcr = None

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CASSETTE_DIR = os.path.join(REPO, "tests", "cassettes")
CASSETTE_NAME = "groq_rca.yaml"

if vcr is not None:
    groq_vcr = vcr.VCR(
        cassette_library_dir=CASSETTE_DIR,
        record_mode="once",
        filter_headers=["authorization", "Authorization"],
        match_on=["method", "scheme", "host", "path", "query", "body"],
    )
else:
    groq_vcr = None


def _cassette_path():
    return os.path.join(CASSETTE_DIR, CASSETTE_NAME)


@pytest.mark.skipif(vcr is None, reason="vcrpy not installed; skipping VCR replay test")
def test_groq_rca_replay(monkeypatch):
    """Replay the recorded Groq RCA exchange (record once, replay offline)."""
    cassette_path = _cassette_path()
    real_key = (os.getenv("GROQ_API_KEY") or "").strip()
    placeholder = "gsk_your_groq_api_key_here"
    has_real_key = bool(real_key) and real_key != placeholder and real_key.startswith("gsk_")

    if not os.path.exists(cassette_path) and not has_real_key:
        pytest.skip(
            "cassette missing and no real GROQ_API_KEY; "
            "record once with APEX_ALLOW_LIVE_LLM=1 and a real key to create it"
        )

    from src.agent.llm_provider import CloudGroqProvider

    # On replay the key value is irrelevant (no network); on the single
    # record run the caller provides the real key explicitly.
    api_key = real_key if has_real_key else "gsk_vcr_replay_dummy_key"
    provider = CloudGroqProvider(api_key=api_key, model_name="openai/gpt-oss-120b")

    system_prompt = "You are a quality RCA assistant. Cite [SOP-SMT-001]."
    user_content = (
        "Line L1: 3 consecutive short_circuit defects. "
        "SOP context: clean stencil per [SOP-SMT-001]."
    )

    with groq_vcr.use_cassette(CASSETTE_NAME):
        text = provider.synthesize_rca(system_prompt, user_content)

    assert isinstance(text, str) and text.strip(), "empty RCA response from cassette"
