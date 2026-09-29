"""
Test correlation-ID middleware.

Bốn hành vi được kiểm, và mỗi cái chặn một lỗi observability khác nhau:

    1. Server SINH ID khi client không gửi            — không có thì log tách rời
    2. Server GIỮ ID hợp lệ do client gửi          — trace xuyên nhiều service
    3. Server TỪ CHỐI ID chứa ký tự lạ              — chống log injection
    4. Log có đủ trường để điều tra                  — timestamp, level, service,
                                                      event, correlation_id
"""
from __future__ import annotations

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _log_lines(capsys) -> list:
    out = capsys.readouterr().out
    return [json.loads(l) for l in out.splitlines() if l.startswith("{")]


# ==========================================================================
# 1. Sinh ID
# ==========================================================================
def test_server_generates_correlation_id_when_absent(client):
    r = client.get("/health")
    assert r.status_code == 200
    cid = r.headers.get("X-Correlation-ID")
    assert cid, "moi response phai co X-Correlation-ID"
    assert len(cid) >= 16, "ID sinh ra phai du dai de la duy nhat"


def test_generated_ids_are_unique(client):
    ids = {client.get("/health").headers["X-Correlation-ID"] for _ in range(5)}
    assert len(ids) == 5, "ID phai khac nhau giua cac request"


# ==========================================================================
# 2. Giữ ID hợp lệ
# ==========================================================================
def test_client_supplied_id_is_preserved(client):
    """Trace phải đi xuyên service nên ID của client phải được giữ nguyên."""
    r = client.get("/health", headers={"X-Correlation-ID": "trace-abc-123"})
    assert r.headers["X-Correlation-ID"] == "trace-abc-123"


def test_id_at_max_length_is_accepted(client):
    cid = "a" * 64
    r = client.get("/health", headers={"X-Correlation-ID": cid})
    assert r.headers["X-Correlation-ID"] == cid


# ==========================================================================
# 3. Chống log injection
# ==========================================================================
@pytest.mark.parametrize("hostile", [
    'fake"}',
    "line1\nline2",
    "tab\there",
    "a" * 200,           # quá dài
    "sp ace/../etc",     # ký tự không thuộc mẫu
])
def test_hostile_correlation_id_is_replaced_not_echoed(client, hostile):
    """
    ID xấu phải bị thay bằng ID do server sinh.

    Nếu ta echo lại chuỗi bất kỳ, kẻ tấn công ghi được dòng log giả — và
    việc điều tra sau đó dựa vào log sẽ tin nhầm.
    """
    r = client.get("/health", headers={"X-Correlation-ID": hostile})
    cid = r.headers["X-Correlation-ID"]
    assert cid != hostile
    assert all(c.isalnum() or c in "._-" for c in cid), \
        f"ID thay thế vẫn chứa ký tự lạ: {cid!r}"


# ==========================================================================
# 4. Log có cấu trúc
# ==========================================================================
def test_structured_log_has_required_fields(client, capsys):
    capsys.readouterr()  # dọn log của request trước
    client.get("/health", headers={"X-Correlation-ID": "log-test-1"})
    entries = _log_lines(capsys)
    assert entries, "phai co it nhat mot dong log JSON"

    e = entries[-1]
    for field in ("timestamp", "level", "service", "event",
                  "correlation_id", "method", "path",
                  "status_code", "duration_ms"):
        assert field in e, f"log thieu truong '{field}'"
    assert e["correlation_id"] == "log-test-1"
    assert e["event"] == "http_request"
    assert e["status_code"] == 200


def test_structured_log_is_valid_json_per_line(client, capsys):
    capsys.readouterr()
    client.get("/health")
    for line in capsys.readouterr().out.splitlines():
        if line.startswith("{"):
            json.loads(line)  # raise nếu dòng log hỏng


def test_log_records_the_status_code_actually_returned(client, capsys):
    """401 phải được ghi là 401, không phải 200 — nếu không, log sẽ phản đối."""
    capsys.readouterr()
    r = client.post("/api/v1/inspections", json={"is_defective": True,
                                                 "inference_time_ms": 10})
    assert r.status_code in (401, 403), "thiếu key phai bi tu choi"
    entries = _log_lines(capsys)
    assert entries[-1]["status_code"] == r.status_code
