import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_smoke.db")

from src.backend.main import app  # noqa: E402

client = TestClient(app)


def test_health_contract_for_render():
    """Render healthCheckPath=/health phải trả 200 với contract ổn định."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "apexinspect-gateway"
    assert "version" in body


def test_static_dashboard_contract_remote():
    """Dashboard tĩnh remote: trỏ API Render + có cấu hình API base/key qua localStorage."""
    src = open("dashboard/app.js", encoding="utf-8").read()
    assert "https://apexinspect-api.onrender.com" in src
    assert "localStorage" in src
    assert "/api/v1/inspections" in src
    assert "/api/v1/mes/action" in src
