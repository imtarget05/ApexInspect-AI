"""Test suite for the LLM gateway. Uses an in-process fake upstream - no network, no LAN."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402

REQUIRED_FIELDS = {
    "ts", "project", "model", "endpoint", "latency_ms",
    "prompt_chars", "prompt_sha256", "completion_chars",
    "usage", "status", "error",
}

SECRET = "TOP-SECRET-PROMPT-CONTENT-9f3a2b"


# ---------------------------------------------------------------- fake upstream


class FakeUpstream:
    """Minimal OpenAI-shaped upstream, plus switches to simulate bad responses."""

    def __init__(self) -> None:
        self.mode = "ok"
        self.calls: list[str] = []
        self.model_ids = ["upstream-model-a"]

    async def app(self, scope, receive, send) -> None:
        assert scope["type"] == "http"
        path = scope["path"]
        self.calls.append(path)
        await receive()

        if self.mode == "400":
            await self._json(send, 400, {"error": {"message": "context length exceeded"}})
            return
        if self.mode == "500":
            await self._json(send, 500, {"error": {"message": "upstream exploded"}})
            return

        if path == "/v1/models":
            await self._json(send, 200, {"object": "list", "data": [
                {"id": mid, "object": "model"} for mid in self.model_ids
            ]})
        elif path == "/v1/chat/completions":
            await self._json(send, 200, {
                "id": "cmpl-1",
                "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            })
        elif path == "/v1/embeddings":
            await self._json(send, 200, {
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            })
        else:
            await self._json(send, 404, {"error": {"message": f"no route {path}"}})

    @staticmethod
    async def _json(send, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


class UnreachableTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        raise httpx.ConnectError(f"connection refused: {request.url.host}", request=request)


class TimeoutTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        raise httpx.ReadTimeout("upstream timed out", request=request)


class RoutingTransport(httpx.AsyncBaseTransport):
    """Dispatches by host so we can have live, dead and timing-out upstreams at once."""

    def __init__(self, routes: dict[str, httpx.AsyncBaseTransport]) -> None:
        self.routes = routes

    async def handle_async_request(self, request):
        transport = self.routes.get(request.url.host)
        if transport is None:
            raise httpx.ConnectError(f"connection refused: {request.url.host}", request=request)
        return await transport.handle_async_request(request)


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def fake() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "llm-telemetry.jsonl"


def build_client(monkeypatch, tmp_path, log_path, routes, *, upstream_env="http://live.local/v1",
                 extra_env=None, max_mb="1") -> TestClient:
    env = {
        "LLM_UPSTREAM": upstream_env,
        "LLM_LOG_PATH": str(log_path),
        "LLM_LOG_MAX_MB": max_mb,
        "LLM_CHAT_MODEL": "cfg-chat-model",
        "LLM_EMBED_MODEL": "cfg-embed-model",
    }
    env.update(extra_env or {})
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for k in ("PORT", "GATEWAY_PORT", "GATEWAY_HOST"):
        monkeypatch.delenv(k, raising=False)
    server._load_config()
    client = TestClient(server.app)
    client.__enter__()
    client.app.state.client = httpx.AsyncClient(
        transport=RoutingTransport(routes), timeout=httpx.Timeout(5.0)
    )
    return client


def live_routes(fake: FakeUpstream) -> dict:
    return {"live.local": httpx.ASGITransport(app=fake.app)}


def read_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def gw(monkeypatch, tmp_path, fake, log_path, request):
    client = build_client(monkeypatch, tmp_path, log_path, live_routes(fake))
    def teardown():
        client.__exit__(None, None, None)
        server.telemetry.close()
        server._load_config()  # env has been restored by monkeypatch by now
    request.addfinalizer(teardown)
    return client


@pytest.fixture
def gw_down(monkeypatch, tmp_path, log_path, fake, request):
    """Gateway whose only upstream is dead."""
    routes = {"dead.local": UnreachableTransport()}
    client = build_client(monkeypatch, tmp_path, log_path, routes, upstream_env="http://dead.local/v1")
    def teardown():
        client.__exit__(None, None, None)
        server.telemetry.close()
        server._load_config()
    request.addfinalizer(teardown)
    return client


# ---------------------------------------------------------------- telemetry


def test_chat_completion_writes_telemetry_with_project(gw, fake, log_path):
    r = gw.post("/v1/chat/completions", headers={"X-Project": "MAIA"},
                json={"model": "m1", "messages": [{"role": "user", "content": "hello world"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi there"

    records = read_records(log_path)
    assert len(records) == 1
    rec = records[0]
    assert REQUIRED_FIELDS <= set(rec)          # backward compatible schema
    assert rec["project"] == "MAIA"
    assert rec["model"] == "m1"
    assert rec["endpoint"] == "/v1/chat/completions"
    assert rec["status"] == 200
    assert rec["error"] is None
    assert rec["prompt_chars"] == len("hello world")
    assert rec["completion_chars"] == len("hi there")
    assert rec["usage"]["total_tokens"] == 9
    assert rec["latency_ms"] >= 0


def test_project_defaults_to_unknown(gw, log_path):
    gw.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert read_records(log_path)[0]["project"] == "unknown"


def test_prompt_body_never_appears_in_the_log(gw, log_path):
    gw.post("/v1/chat/completions", headers={"X-Project": "CreditFlow"},
            json={"model": "m", "messages": [{"role": "user", "content": SECRET}]})
    gw.post("/v1/embeddings", headers={"X-Project": "CreditFlow"},
            json={"model": "e", "input": SECRET})

    raw = log_path.read_text(encoding="utf-8")
    assert SECRET not in raw
    for rec in read_records(log_path):
        assert set(rec) >= {"prompt_sha256"} and isinstance(rec["prompt_sha256"], str)
        assert len(rec["prompt_sha256"]) == 16
        assert isinstance(rec["prompt_chars"], int)


def test_embeddings_forwarded_and_logged(gw, log_path):
    r = gw.post("/v1/embeddings", headers={"X-Project": "ApexInspect-AI"},
                json={"model": "e1", "input": "vector me"})
    assert r.status_code == 200
    assert len(r.json()["data"][0]["embedding"]) == 3
    rec = read_records(log_path)[0]
    assert rec["endpoint"] == "/v1/embeddings"
    assert rec["project"] == "ApexInspect-AI"
    assert rec["prompt_chars"] == len("vector me")
    assert rec["completion_chars"] == len(str([0.1, 0.2, 0.3]))  # historical metric


# ---------------------------------------------------------------- upstream errors


def test_upstream_400_passed_through_and_logged_as_error(gw, fake, log_path):
    fake.mode = "400"
    r = gw.post("/v1/chat/completions", headers={"X-Project": "MAIA"},
                json={"model": "m", "messages": [{"role": "user", "content": "too long"}]})
    assert r.status_code == 400                                   # same status to the caller
    assert r.json()["error"]["message"] == "context length exceeded"  # same bytes to the caller

    rec = read_records(log_path)[0]
    assert rec["status"] == 400
    assert rec["error"] is not None                               # was the defect
    assert rec["error"].startswith("upstream_400")
    assert "context length exceeded" in rec["error"]


def test_upstream_500_logged_as_error(gw, fake, log_path):
    fake.mode = "500"
    r = gw.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 500
    assert read_records(log_path)[0]["error"].startswith("upstream_500")


def test_unreachable_upstream_returns_502_and_logs(gw_down, log_path):
    r = gw_down.post("/v1/chat/completions", headers={"X-Project": "IT-Helpdesk-Lab"},
                     json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    rec = read_records(log_path)[0]
    assert rec["status"] == 502
    assert rec["error"]
    assert rec["project"] == "IT-Helpdesk-Lab"


def test_upstream_timeout_returns_502(monkeypatch, tmp_path, fake, log_path, request):
    routes = {"slow.local": TimeoutTransport()}
    client = build_client(monkeypatch, tmp_path, log_path, routes, upstream_env="http://slow.local/v1")
    request.addfinalizer(lambda: (client.__exit__(None, None, None), server.telemetry.close(), server._load_config()))
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 502
    assert read_records(log_path)[0]["status"] == 502


# ---------------------------------------------------------------- health / readiness


def test_health_is_200_even_when_upstream_is_down(gw_down):
    r = gw_down.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert set(body) == {"status", "upstream", "time"}
    assert set(body["upstream"]) == {"url", "reachable", "latency_ms", "error"}
    assert body["upstream"]["reachable"] is False
    assert body["upstream"]["url"] == "http://dead.local/v1"
    assert body["upstream"]["error"]


def test_health_ok_when_upstream_up(gw):
    body = gw.get("/health").json()
    assert body["status"] == "ok"
    assert body["upstream"]["reachable"] is True
    assert body["upstream"]["latency_ms"] >= 0
    assert body["upstream"]["error"] is None


def test_ready_503_when_upstream_down(gw_down):
    r = gw_down.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"
    assert r.json()["upstream"]["reachable"] is False


def test_ready_200_when_upstream_up(gw):
    r = gw.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["upstream"]["reachable"] is True


def test_probe_result_is_cached(fake, gw):
    gw.get("/health")
    gw.get("/health")
    gw.get("/health/ready")
    assert fake.calls.count("/v1/models") == 1     # TTL cache, not one probe per call


# ---------------------------------------------------------------- fallback chain


def test_fallback_chain_picks_reachable_upstream(monkeypatch, tmp_path, fake, log_path, request):
    routes = {
        "dead.local": UnreachableTransport(),
        "live.local": httpx.ASGITransport(app=fake.app),
    }
    chain = "http://dead.local/v1,http://live.local/v1"
    client = build_client(monkeypatch, tmp_path, log_path, routes, upstream_env=chain)
    request.addfinalizer(lambda: (client.__exit__(None, None, None), server.telemetry.close(), server._load_config()))

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["upstream"]["url"] == "http://live.local/v1"

    r = client.post("/v1/chat/completions", headers={"X-Project": "Factory-Data-Automation"},
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    rec = read_records(log_path)[0]
    assert rec["status"] == 200
    assert rec["upstream"] == "http://live.local/v1"


def test_fallback_is_sticky(monkeypatch, tmp_path, fake, log_path, request):
    routes = {
        "a.local": httpx.ASGITransport(app=fake.app),
        "b.local": httpx.ASGITransport(app=fake.app),
    }
    client = build_client(monkeypatch, tmp_path, log_path, routes, upstream_env="http://a.local/v1,http://b.local/v1")
    request.addfinalizer(lambda: (client.__exit__(None, None, None), server.telemetry.close(), server._load_config()))
    client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert server._sticky_index == 0
    server._sticky_index = 1
    client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert read_records(log_path)[-1]["upstream"] == "http://b.local/v1"  # stayed on b


def test_chain_entry_without_v1_is_normalised(monkeypatch, tmp_path, log_path, request):
    build_client(monkeypatch, tmp_path, log_path, {}, upstream_env="192.168.1.8:1234,http://x.local/api")
    assert server.UPSTREAMS == ["http://192.168.1.8:1234/v1", "http://x.local/v1"]
    request.addfinalizer(lambda: (server.telemetry.close(), server._load_config()))


def test_all_upstreams_down_returns_502(monkeypatch, tmp_path, log_path, request):
    client = build_client(monkeypatch, tmp_path, log_path, {},
                          upstream_env="http://dead1.local/v1,http://dead2.local/v1")
    request.addfinalizer(lambda: (client.__exit__(None, None, None), server.telemetry.close(), server._load_config()))
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 502
    assert read_records(log_path)[0]["status"] == 502


# ---------------------------------------------------------------- models


def test_models_includes_configured_chat_and_embed_ids(gw):
    body = gw.get("/v1/models").json()
    ids = [m["id"] for m in body["data"]]
    assert "upstream-model-a" in ids          # from upstream
    assert "cfg-chat-model" in ids           # from LLM_CHAT_MODEL
    assert "cfg-embed-model" in ids          # from LLM_EMBED_MODEL


def test_models_does_not_duplicate_existing_ids(gw, fake):
    fake.model_ids = ["cfg-chat-model", "cfg-embed-model"]
    body = gw.get("/v1/models").json()
    ids = [m["id"] for m in body["data"]]
    assert ids.count("cfg-chat-model") == 1
    assert len(ids) == 2


def test_models_502_when_upstream_down(gw_down):
    r = gw_down.get("/v1/models")
    assert r.status_code == 502


# ---------------------------------------------------------------- log rotation


def test_log_rotation_keeps_exactly_one_backup(monkeypatch, tmp_path, fake, log_path, request):
    client = build_client(monkeypatch, tmp_path, log_path, live_routes(fake), max_mb="0.0004")
    request.addfinalizer(lambda: (client.__exit__(None, None, None), server.telemetry.close(), server._load_config()))
    for _ in range(25):
        client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "rotate me"}]})
    backup = log_path.with_name(log_path.name + ".1")
    assert backup.exists(), "expected rotation to llm-telemetry.jsonl.1"
    assert not log_path.with_name(log_path.name + ".2").exists(), "only 1 backup is kept"
    assert log_path.stat().st_size < log_path.stat().st_size + backup.stat().st_size
    # the active file stayed near the limit instead of growing without bound
    assert log_path.stat().st_size < 25 * 420
    # both files are still valid JSONL, and the kept backup holds the older records
    older, newer = read_records(backup), read_records(log_path)
    assert older and newer
    assert older[0]["ts"] <= newer[-1]["ts"]
    assert len(older) + len(newer) < 25, "rotation dropped history it should have kept"


def test_no_rotation_under_limit(gw, log_path):
    for _ in range(3):
        gw.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "small"}]})
    assert not log_path.with_name(log_path.name + ".1").exists()
    assert len(read_records(log_path)) == 3


# ---------------------------------------------------------------- config


def test_port_is_canonical_over_deprecated_gateway_port(monkeypatch):
    monkeypatch.setenv("PORT", "9001")
    monkeypatch.setenv("GATEWAY_PORT", "7000")
    server._load_config()
    assert server.PORT == 9001


def test_gateway_port_used_when_port_absent(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("GATEWAY_PORT", "7000")
    server._load_config()
    assert server.PORT == 7000


def test_port_defaults_to_8787(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("GATEWAY_PORT", raising=False)
    server._load_config()
    assert server.PORT == 8787


def test_host_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("GATEWAY_HOST", raising=False)
    server._load_config()
    assert server.HOST == "127.0.0.1"
    monkeypatch.setenv("GATEWAY_HOST", "0.0.0.0")
    server._load_config()
    assert server.HOST == "0.0.0.0"
    monkeypatch.delenv("GATEWAY_HOST", raising=False)


def test_log_max_mb_default_is_10(monkeypatch):
    monkeypatch.delenv("LLM_LOG_MAX_MB", raising=False)
    server._load_config()
    assert server.LOG_MAX_BYTES == 10 * 1024 * 1024


# ---------------------------------------------------------------- no native Ollama proxy


def test_ollama_native_api_is_not_proxied(gw, fake):
    for path in ("/api/embeddings", "/api/chat", "/api/generate"):
        assert gw.post(path, json={"model": "m", "prompt": "x"}).status_code == 404
    assert fake.calls == []

def test_metrics_endpoint_returns_prometheus_format(gw):
    resp = gw.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    # Call chat completion to bump metrics
    gw.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Project": "TestProject"},
    )
    resp2 = gw.get("/metrics")
    assert resp2.status_code == 200
    assert "llm_requests_total" in resp2.text
    assert 'project="TestProject"' in resp2.text


def test_admin_stats_calculates_aggregations(gw):
    gw.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Project": "StatsProj"},
    )
    resp = gw.get("/admin/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "projects" in data
    assert "StatsProj" in data["projects"]
    proj_stat = data["projects"]["StatsProj"]
    assert proj_stat["requests"] >= 1
    assert "latency_p50_ms" in proj_stat
    assert "est_commercial_cost_saved_usd" in proj_stat


def test_model_alias_resolution(gw, fake):
    # Call using alias 'reasoning'
    gw.post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "solve this"}]},
        headers={"X-Project": "MAIA"},
    )
    log_rec = gw.app.state.last_rec if hasattr(gw.app.state, "last_rec") else None
    # Verify alias mapping function directly
    assert server.resolve_model_alias("reasoning") == "deepseek/deepseek-r1-0528-qwen3-8b"
    assert server.resolve_model_alias("fast") == "nvidia/nemotron-3-nano-4b"
    assert server.resolve_model_alias("vision") == "qwen2.5-vl-3b-instruct"
    # Auto-detect image
    img_msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://img.jpg"}}]}]
    assert server.resolve_model_alias("", has_image=True) == "qwen2.5-vl-3b-instruct"


# ---------------------------------------------------------------- v2: routing / resilience / governance


def test_xtask_header_overrides_model_alias(gw, log_path):
    r = gw.post("/v1/chat/completions", headers={"X-Project": "MAIA", "X-Task": "reasoning"},
                json={"model": "fast", "messages": [{"role": "user", "content": "solve this"}]})
    assert r.status_code == 200
    rec = read_records(log_path)[-1]
    assert rec["model"] == "deepseek/deepseek-r1-0528-qwen3-8b"
    assert rec["requested_model"] == "fast"
    assert rec["task"] == "reasoning"


def test_model_env_override(monkeypatch, tmp_path, fake, log_path, request):
    routes = live_routes(fake)
    client = build_client(monkeypatch, tmp_path, log_path, routes,
                          extra_env={"LLM_MODEL_FAST": "custom/nemotron-override",
                                     "LLM_RETRY_BASE_MS": "1"})
    request.addfinalizer(lambda: (client.__exit__(None, None, None), server.telemetry.close(), server._load_config()))
    assert server.MODEL_ALIASES["fast"] == "custom/nemotron-override"
    assert server.resolve_model_alias("fast") == "custom/nemotron-override"


def test_pii_masking_redacts_vn_phone_and_cccd(gw, log_path):
    phone, cccd = "0901234567", "001234567890"
    masked, n = server._mask_pii_in_text(f"gọi {phone} cccd {cccd} nhé")
    assert n >= 2 and phone not in masked and cccd not in masked

    r = gw.post("/v1/chat/completions", headers={"X-Project": "CreditFlow"},
                json={"model": "m", "messages": [{"role": "user", "content": f"sđt {phone} cccd {cccd}"}]})
    assert r.status_code == 200
    raw = log_path.read_text(encoding="utf-8")
    assert phone not in raw and cccd not in raw
    rec = read_records(log_path)[-1]
    assert rec["pii_redactions"] >= 2


def test_circuit_breaker_opens_and_failfast(monkeypatch, tmp_path, log_path, request):
    server._reset_circuit_state()
    routes = {"dead.local": UnreachableTransport()}
    client = build_client(monkeypatch, tmp_path, log_path, routes,
                          upstream_env="http://dead.local/v1",
                          extra_env={"LLM_CB_FAILURE_THRESHOLD": "2", "LLM_MAX_RETRIES": "0",
                                     "LLM_RETRY_BASE_MS": "1"})
    request.addfinalizer(lambda: (client.__exit__(None, None, None), server.telemetry.close(),
                                  server._load_config(), server._reset_circuit_state()))
    assert client.post("/v1/chat/completions", json={"model": "m", "messages": []}).status_code == 502
    assert client.post("/v1/chat/completions", json={"model": "m", "messages": []}).status_code == 502
    assert server._cb_state["http://dead.local/v1"]["state"] == "open"
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 502 and "circuit_open" in r.json()["error"]["message"]
    assert "llm_circuit_breaker_open" in client.get("/metrics").text


def test_retry_recovers_from_transient_503(monkeypatch, tmp_path, log_path, request):
    class FlakyTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.n = 0

        async def handle_async_request(self, req):
            self.n += 1
            if self.n == 1:
                return httpx.Response(503, json={"error": {"message": "busy"}})
            return httpx.Response(200, json={
                "id": "cmpl-1",
                "choices": [{"message": {"role": "assistant", "content": "recovered"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

    server._reset_circuit_state()
    flaky = FlakyTransport()
    client = build_client(monkeypatch, tmp_path, log_path, {"flaky.local": flaky},
                          upstream_env="http://flaky.local/v1",
                          extra_env={"LLM_MAX_RETRIES": "3", "LLM_RETRY_BASE_MS": "1"})
    request.addfinalizer(lambda: (client.__exit__(None, None, None), server.telemetry.close(),
                                  server._load_config(), server._reset_circuit_state()))
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "recovered"
    assert flaky.n == 2  # one 503 + one retry that succeeded


def test_quota_denied_when_exceeded(monkeypatch, tmp_path, fake, log_path, request):
    server._reset_quota_state()
    server._reset_circuit_state()
    client = build_client(monkeypatch, tmp_path, log_path, live_routes(fake),
                          extra_env={"LLM_QUOTA_TOKENS_PER_DAY": "10", "LLM_RETRY_BASE_MS": "1"})
    request.addfinalizer(lambda: (client.__exit__(None, None, None), server.telemetry.close(),
                                  server._load_config(), server._reset_quota_state(),
                                  server._reset_circuit_state()))
    first = client.post("/v1/chat/completions", headers={"X-Project": "QuotaProj"},
                        json={"model": "m", "messages": [{"role": "user", "content": "hello world"}]})
    assert first.status_code == 200  # fake usage.total_tokens = 9 <= 10
    denied = client.post("/v1/chat/completions", headers={"X-Project": "QuotaProj"},
                         json={"model": "m", "messages": [{"role": "user", "content": "hello again"}]})
    assert denied.status_code == 429
    assert denied.json()["error"]["type"] == "quota_exceeded"
    q = client.get("/v1/quotas").json()
    assert q["quota_tokens_per_day"] == 10 and q["usage_tokens"]["QuotaProj"] == 9
    assert "llm_quota_denied_total" in client.get("/metrics").text


def test_streaming_passthrough_path(gw, log_path):
    r = gw.post("/v1/chat/completions", headers={"X-Project": "MAIA"},
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert r.headers.get("X-Request-Id")
    rec = read_records(log_path)[-1]
    assert rec.get("stream") is True


def test_routing_and_quotas_endpoints(gw):
    routing = gw.get("/v1/routing").json()
    assert routing["aliases"]["reasoning"] == "deepseek/deepseek-r1-0528-qwen3-8b"
    assert routing["aliases"]["vision"] == "qwen2.5-vl-3b-instruct"
    assert routing["aliases"]["embed"] == "text-embedding-nomic-embed-text-v1.5"
    assert routing["vision_auto_detect"] is True
    quotas = gw.get("/v1/quotas").json()
    assert set(quotas) >= {"date", "quota_tokens_per_day", "usage_tokens"}


def test_cloud_failover_when_lan_down(monkeypatch, tmp_path, fake, log_path, request):
    server._reset_circuit_state()
    server._reset_quota_state()
    routes = {
        "dead.local": UnreachableTransport(),
        "cloud.example": httpx.ASGITransport(app=fake.app),
    }
    client = build_client(
        monkeypatch, tmp_path, log_path, routes,
        upstream_env="http://dead.local/v1",
        extra_env={"LLM_CLOUD_UPSTREAM": "http://cloud.example/v1",
                   "LLM_CLOUD_MODEL": "cloud-llama",
                   "LLM_MAX_RETRIES": "0", "LLM_RETRY_BASE_MS": "1"},
    )
    request.addfinalizer(lambda: (client.__exit__(None, None, None), server.telemetry.close(),
                                  server._load_config(), server._reset_circuit_state(),
                                  server._reset_quota_state()))
    assert server.CLOUD_UPSTREAM == "http://cloud.example/v1"
    r = client.post("/v1/chat/completions", headers={"X-Project": "MAIA"},
                    json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    rec = read_records(log_path)[-1]
    assert rec["upstream"] == "http://cloud.example/v1"
    assert "llm_cloud_requests_total" in client.get("/metrics").text
    assert client.get("/v1/routing").json()["cloud_failover"] is True

