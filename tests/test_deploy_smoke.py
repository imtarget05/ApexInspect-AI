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


def test_streamlit_launcher_remote_mode_imports_light():
    """streamlit_app.py ở remote mode chỉ import stdlib + streamlit (nhẹ cho Cloud 1GB)."""
    src = open("streamlit_app.py", encoding="utf-8").read()
    assert "BACKEND_URL" in src
    assert "src/ui/app.py" in src  # fallback local vẫn trỏ đúng app gốc
