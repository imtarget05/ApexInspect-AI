"""Regression guard: the test suite must never reach api.groq.com.

Root cause #2 (measured, not inferred): `RCAAnalysisAgent` read
`os.getenv("GROQ_API_KEY")` *after* `load_dotenv()` pulled the real key out of
`.env`, so every test that triggered the incident cascade made a live HTTPS
call. A socket-level sniffer recorded 9 external connects to
`2a06:98c1:310a::ac40:9514`, which resolves to `api.groq.com` — that is what
made the suite take 49.61s on one run and 7.35s on the next, and it is the
network dependency behind the flaky failures.

The fix is a single gate (`src/runtime_flags.effective_groq_key`) enforced in
production code, so it holds for every runner (pytest and
`python3 -m unittest discover tests/`), not just for pytest's conftest hook.
"""
import os
import re
import subprocess
import sys
import textwrap

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
FAKE_KEY = "gsk_regression_fake_key_should_never_leave_the_process"


def _child(code, extra_env=None):
    env = dict(os.environ)
    env["GROQ_API_KEY"] = FAKE_KEY
    env.pop("APEX_ALLOW_LIVE_LLM", None)
    env.pop("APEX_ALLOW_LIVE_LLM_IN_TESTS", None)
    env.update(extra_env or {})
    return subprocess.run(
        [PY, "-c", textwrap.dedent(code)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )


def test_key_is_blanked_inside_a_test_runner():
    """A real-looking key in the environment must not survive in a test run."""
    r = _child("""
        import pytest  # marks this interpreter as a test runner
        from src.runtime_flags import effective_groq_key, is_test_process
        print("TEST_PROCESS", is_test_process())
        print("KEY", repr(effective_groq_key()))
    """)
    assert r.returncode == 0, r.stderr
    assert "TEST_PROCESS True" in r.stdout, r.stdout
    assert "KEY ''" in r.stdout, f"key leaked into test process: {r.stdout}"


def test_live_llm_opt_in_restores_the_key():
    """The gate must be escapable on purpose, not a silent behaviour change."""
    r = _child("""
        import pytest
        from src.runtime_flags import effective_groq_key
        print("KEY", repr(effective_groq_key()))
    """, {"APEX_ALLOW_LIVE_LLM": "1"})
    assert r.returncode == 0, r.stderr
    assert f"KEY '{FAKE_KEY}'" in r.stdout, r.stdout


def test_rca_agent_never_reads_the_key_directly():
    """The agent must obtain the key through the gate, not from os.getenv."""
    path = os.path.join(REPO, "src", "agent", "subagents", "rca_analysis_agent.py")
    src = open(path, encoding="utf-8").read()
    for pattern in ('os.getenv("GROQ_API_KEY")',
                    "os.getenv('GROQ_API_KEY')",
                    'os.environ["GROQ_API_KEY"]',
                    "os.environ.get(\"GROQ_API_KEY\")"):
        assert pattern not in src, f"ungated key read still present: {pattern}"
    assert "effective_groq_key" in src, "agent does not use the gated key helper"


def test_rca_agent_end_to_end_offline():
    """End-to-end proof: the RCA sub-agent cannot dial Groq during tests."""
    r = _child("""
        import pytest
        from src.agent.subagents.rca_analysis_agent import RCAAnalysisAgent

        agent = RCAAnalysisAgent()
        print("AGENT_KEY", repr(agent.groq_api_key))
        result = agent.run({
            "defect_class": "short_circuit",
            "consecutive_count": 3,
            "line_id": "L1",
            "sop_context": "",
            "sop_citations": ["SOP-SMT-001"],
        })
        text = result["rca_analysis"]
        print("RCA_IS_FALLBACK", "SOP-SMT-001" in text and "HALT_LINE" in text)
    """)
    assert r.returncode == 0, r.stderr
    assert "AGENT_KEY ''" in r.stdout, r.stdout
    assert "RCA_IS_FALLBACK True" in r.stdout, r.stdout


def test_orchestrator_holds_no_key_inside_a_test_runner():
    """The LangGraph orchestrator (backend entrypoint) must be offline in tests."""
    r = _child("""
        import pytest
        from src.agent.graph import QualityIncidentAgent

        orchestrator = QualityIncidentAgent()
        print("ORCH_KEY", repr(orchestrator.groq_api_key))
        print("RCA_AGENT_KEY", repr(orchestrator.rca_agent.groq_api_key))
    """)
    assert r.returncode == 0, r.stderr
    assert "ORCH_KEY ''" in r.stdout, r.stdout
    assert "RCA_AGENT_KEY ''" in r.stdout, r.stdout


def test_no_ungated_groq_key_reads_anywhere_in_src():
    """Only the gate module may mention GROQ_API_KEY, so the fix cannot rot."""
    offenders = []
    for root, _dirs, files in os.walk(os.path.join(REPO, "src")):
        for name in files:
            if not name.endswith(".py") or name == "runtime_flags.py":
                continue
            path = os.path.join(root, name)
            text = open(path, encoding="utf-8").read()
            if re.search(r"GROQ_API_KEY", text):
                offenders.append(os.path.relpath(path, REPO))
    assert offenders == [], f"GROQ_API_KEY read outside the gate: {offenders}"
