import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend import main as backend_main

VALID_KEY = "unit-test-factory-key-0123456789"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", VALID_KEY)
    monkeypatch.setenv("APP_ENV", "development")
    return TestClient(backend_main.app)


def _headers(key=VALID_KEY):
    return {"X-API-KEY": key}


# --- C1: no hardcoded default, fail closed outside development --------------


def _main_source():
    return open(
        os.path.join(os.path.dirname(__file__), "..", "src", "backend", "main.py"),
        encoding="utf-8",
    ).read()


def test_no_committed_default_api_key():
    """The literal that used to be the fallback must be gone from the source.

    Checked on the parsed AST rather than the raw text, so the explanatory
    docstrings that name the old pattern do not trip the check.
    """
    import ast

    src = _main_source()
    assert "dev-factory-key-secret" not in src

    defaults = []
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "getenv"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "API_KEY"
        ):
            default = node.args[1] if len(node.args) > 1 else None
            defaults.append(ast.literal_eval(default) if default is not None else None)

    assert defaults, "API_KEY should still be read from the environment"
    for value in defaults:
        assert value in (None, ""), f"API_KEY must have no default value, got {value!r}"


def test_missing_api_key_raises_outside_development(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError, match="API_KEY"):
        backend_main._expected_api_key()


def test_unset_api_key_mints_an_ephemeral_key_in_development(monkeypatch):
    """No committed default, but local work still functions."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setattr(backend_main, "_ephemeral_api_key", None)
    key = backend_main._expected_api_key()
    assert key and "dev-factory" not in key
    assert backend_main._expected_api_key() == key  # stable within the process


def test_empty_api_key_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("API_KEY", "   ")
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError, match="API_KEY"):
        backend_main._expected_api_key()


def test_legacy_disable_sentinels_no_longer_open_the_gate(monkeypatch):
    """API_KEY=none/development/disabled used to disable verification entirely.

    A sentinel is now just an (absurd) key: it is compared against, not
    treated as "auth disabled", so no header still fails closed.
    """
    monkeypatch.setenv("APP_ENV", "production")
    for sentinel in ("none", "development", "disabled", "NONE"):
        monkeypatch.setenv("API_KEY", sentinel)
        assert backend_main._expected_api_key() == sentinel

        with pytest.raises(backend_main.HTTPException) as exc:
            backend_main.verify_api_key(api_key=None)
        assert exc.value.status_code == 401

        with pytest.raises(backend_main.HTTPException) as exc:
            backend_main.verify_api_key(api_key="anything")
        assert exc.value.status_code == 403



def test_comparison_is_constant_time():
    src = _main_source()
    assert "hmac.compare_digest" in src
    assert "api_key == expected" not in src


# --- C2: the telemetry write and the read endpoints are protected ----------


def test_inspections_requires_api_key(client):
    payload = {
        "line_id": "SMT-LINE-01",
        "is_defective": True,
        "defect_classes": ["short_circuit"],
        "confidence_scores": [0.9],
        "bounding_boxes": [[1, 1, 10, 10]],
        "inference_time_ms": 25.0,
    }
    assert client.post("/api/v1/inspections", json=payload).status_code == 401
    assert client.post(
        "/api/v1/inspections", json=payload, headers=_headers("wrong")
    ).status_code == 403


def test_inspections_accepts_the_valid_key(client):
    payload = {
        "line_id": "SMT-LINE-01",
        "is_defective": True,
        "defect_classes": ["short_circuit"],
        "confidence_scores": [0.9],
        "bounding_boxes": [[1, 1, 10, 10]],
        "inference_time_ms": 25.0,
    }
    res = client.post("/api/v1/inspections", json=payload, headers=_headers())
    assert res.status_code == 200, res.text


def test_public_caller_cannot_drive_the_halt_line_trigger(client):
    """Three anonymous defective posts must not open a PENDING_APPROVAL ticket.

    This is the concrete impact: the 3-consecutive-defect trigger creates a
    ticket whose approval dispatches a real HALT_LINE actuation.
    """
    payload = {
        "line_id": "SMT-LINE-99",
        "is_defective": True,
        "defect_classes": ["short_circuit"],
        "confidence_scores": [0.99],
        "bounding_boxes": [[1, 1, 10, 10]],
        "inference_time_ms": 25.0,
    }
    for _ in range(5):
        assert client.post("/api/v1/inspections", json=payload).status_code == 401

    # Nothing was written, so the trigger cannot have fired.
    from src.backend.database import SessionLocal
    from src.backend.models import InspectionLog, MESTicket

    db = SessionLocal()
    try:
        assert db.query(InspectionLog).filter_by(line_id="SMT-LINE-99").count() == 0
        assert db.query(MESTicket).filter_by(line_id="SMT-LINE-99").count() == 0
    finally:
        db.close()


def test_line_metrics_requires_api_key(client):
    assert client.get("/api/v1/lines/SMT-LINE-01/metrics").status_code == 401
    assert client.get(
        "/api/v1/lines/SMT-LINE-01/metrics", headers=_headers("wrong")
    ).status_code == 403


def test_recent_tickets_requires_api_key(client):
    assert client.get("/api/v1/tickets/recent").status_code == 401
    assert client.get("/api/v1/tickets/recent", headers=_headers("wrong")).status_code == 403


def test_health_endpoints_stay_open(client):
    """Liveness/readiness must not require a key or the probe breaks."""
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


# --- C3: CORS is an explicit allowlist -------------------------------------


def test_wildcard_origin_is_rejected(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        backend_main._resolve_cors_origins()


def test_origins_are_parsed_explicitly(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example , https://b.example/")
    assert backend_main._resolve_cors_origins() == [
        "https://a.example",
        "https://b.example",
    ]


def test_default_is_same_origin_only(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert backend_main._resolve_cors_origins() == []


def test_no_wildcard_with_credentials_in_source():
    """No CORS keyword may be given a literal "*" alongside credentials.

    `allow_origins` is now the explicit list returned by
    `_resolve_cors_origins()`, which rejects "*" outright (tested above); this
    checks no other CORS keyword reintroduces a wildcard literal.
    """
    import ast

    tree = ast.parse(_main_source())
    seen_origins = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_middleware":
            continue
        for kw in node.keywords:
            if kw.arg == "allow_origins":
                seen_origins = True
            if isinstance(kw.value, ast.List):
                values = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
                assert "*" not in values, f"{kw.arg} must not contain a wildcard"
    assert seen_origins, "expected an explicit allow_origins to be configured"
