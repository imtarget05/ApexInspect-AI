"""LLM Gateway - OpenAI-compatible proxy to a LAN LLM (LM Studio / Ollama) + centralized telemetry.

Contract (stable - other repos code against this):
  Base URL ............ http://localhost:8787/v1
  GET  /health ........ always HTTP 200 while the process is alive
  GET  /health/ready ... 200 ready / 503 not_ready (upstream probe)
  GET  /v1/models ..... upstream list + configured LLM_CHAT_MODEL / LLM_EMBED_MODEL
  POST /v1/chat/completions (unary JSON or SSE when {"stream": true})
  POST /v1/embeddings
  GET  /metrics ....... Prometheus text exposition (zero extra deps)
  GET  /v1/routing .... alias -> concrete model map + policy
  GET  /v1/quotas ..... per-project daily token usage vs quota
  GET  /admin/stats ... aggregated telemetry (p50/p95, error rate, cost saved)
  Headers: X-Project (accounting) | X-Task (fast|reasoning|vision|vision-fast|embed|general|long|auto)

Only OpenAI shapes are exposed. Ollama-native /api/* is intentionally NOT proxied.

Telemetry: one JSONL line per forwarded request in LLM_LOG_PATH. Prompt bodies are
never stored - only character counts and a truncated sha256.

Run: python server.py   (PORT, or the deprecated GATEWAY_PORT alias)
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import threading
import time
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from metrics import metrics_registry

# ---------------------------------------------------------------- config

DEFAULT_UPSTREAM = "http://192.168.1.8:1234/v1"
DEFAULT_LOG_PATH = Path(__file__).parent / "logs" / "llm-telemetry.jsonl"
DEFAULT_PORT = 8787
DEFAULT_HOST = "127.0.0.1"
DEFAULT_TIMEOUT = 120.0
DEFAULT_LOG_MAX_MB = 10.0
PROBE_TTL_S = 5.0
PROBE_TIMEOUT_S = 3.0
ERR_SNIPPET_CHARS = 200

# Task-based aliases mapped to the 7 LM Studio models on the LAN host.
# Overridable per slot via LLM_MODEL_<SLOT> env (see _load_config).
MODEL_ALIASES: dict[str, str] = {}
DEFAULT_ALIASES: dict[str, str] = {
    "reasoning": "deepseek/deepseek-r1-0528-qwen3-8b",
    "deepseek": "deepseek/deepseek-r1-0528-qwen3-8b",
    "fast": "nvidia/nemotron-3-nano-4b",
    "nemotron": "nvidia/nemotron-3-nano-4b",
    "code": "nvidia/nemotron-3-nano-4b",
    "general": "google/gemma-4-e4b",
    "gemma": "google/gemma-4-e4b",
    "long": "google/gemma-4-e4b",
    "vision": "qwen2.5-vl-3b-instruct",
    "qwen-vl": "qwen2.5-vl-3b-instruct",
    "vision-fast": "zai-org/glm-4.6v-flash",
    "glm": "zai-org/glm-4.6v-flash",
    "embed": "text-embedding-nomic-embed-text-v1.5",
    "nomic": "text-embedding-nomic-embed-text-v1.5",
}
# Slots that map 1:1 to env names LLM_MODEL_<SLOT>.
_ALIAS_SLOTS = ("FAST", "REASONING", "VISION", "VISION_ALT", "EMBED", "GENERAL", "LONG")

# PII patterns (Vietnam): mobile +84/0[3,5,7,8,9] + CCCD 12 digits (+ CMND 9 digits).
PII_PHONE_RE = re.compile(r"(?:\+84[\s.\-]*|0)(?:3|5|7|8|9)[\s.\-]*\d[\s.\-]*\d[\s.\-]*\d[\s.\-]*\d[\s.\-]*\d[\s.\-]*\d[\s.\-]*\d[\s.\-]*\d\b")
PII_CCCD_RE = re.compile(r"\b\d{12}\b")
PII_CMND_RE = re.compile(r"(?<![\d])\d{9}(?![\d])")
PII_TOKEN = "[REDACTED_PII]"

RETRYABLE_STATUSES = {429, 502, 503, 504}


UPSTREAMS: list[str] = []
LOG_PATH: Path = DEFAULT_LOG_PATH
TIMEOUT: float = DEFAULT_TIMEOUT
LOG_MAX_BYTES: int = int(DEFAULT_LOG_MAX_MB * 1024 * 1024)
PORT: int = DEFAULT_PORT
HOST: str = DEFAULT_HOST
CHAT_MODEL: str = ""
EMBED_MODEL: str = ""
# Resilience / governance knobs (env-overridable, zero-cost defaults).
CB_FAILURE_THRESHOLD: int = 5
CB_OPEN_SECONDS: float = 30.0
MAX_RETRIES: int = 2
RETRY_BASE_MS: float = 50.0
PII_MASKING: bool = True
QUOTA_TOKENS_PER_DAY: int = 0  # 0 = unlimited
COST_PER_1M_USD: float = 10.0
CLOUD_UPSTREAM: str = ""  # optional OpenAI-compatible cloud base (hybrid failover)
CLOUD_API_KEY: str = ""
CLOUD_MODEL: str = ""  # concrete cloud model id used when failing over (else keep LAN id)
CLOUD_MODEL_REMAP: bool = True  # rewrite LAN model id to CLOUD_MODEL when failing over

_sticky_index: int | None = None
_probe_cache: dict[str, Any] = {"at": 0.0, "result": None}
# Circuit breaker per upstream base: {"failures": int, "state": str, "opened_at": float}
_cb_state: dict[str, dict[str, Any]] = {}
# Quota usage per (project, yyyy-mm-dd): tokens
_quota_usage: dict[tuple[str, str], int] = {}
# Guards CB + quota read-modify-write against threaded servers / free-threaded runtimes.
_state_lock = threading.Lock()


def _normalise_upstream(url: str) -> str:
    """Accept 'host:port', 'http://host:port', '.../v1' or '.../api' and return a
    base that is safe to append '/chat/completions' to."""
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    if not re.match(r"^https?://", u):
        u = "http://" + u
    if re.search(r"/v1$", u):
        return u
    if re.search(r"/api$", u):  # Ollama-native base -> its OpenAI-compatible base
        u = u[: -len("/api")] + "/v1"
    else:
        u = u + "/v1"
    return u


def _reset_upstream_state() -> None:
    global _sticky_index
    _sticky_index = None


def _reset_probe_cache() -> None:
    global _probe_cache
    _probe_cache = {"at": 0.0, "result": None}


def _reset_circuit_state() -> None:
    with _state_lock:
        _cb_state.clear()


def _reset_quota_state() -> None:
    with _state_lock:
        _quota_usage.clear()


def _build_aliases() -> dict[str, str]:
    """DEFAULT_ALIASES with per-slot env overrides (LLM_MODEL_FAST, ...)."""
    m = dict(DEFAULT_ALIASES)
    slot_default = {
        "FAST": m["fast"],
        "REASONING": m["reasoning"],
        "VISION": m["vision"],
        "VISION_ALT": m["vision-fast"],
        "EMBED": m["embed"],
        "GENERAL": m["general"],
        "LONG": m["long"],
    }
    for slot in _ALIAS_SLOTS:
        override = os.environ.get(f"LLM_MODEL_{slot}", "").strip()
        if override:
            slot_default[slot] = override
    m.update({
        "fast": slot_default["FAST"], "nemotron": slot_default["FAST"], "code": slot_default["FAST"],
        "reasoning": slot_default["REASONING"], "deepseek": slot_default["REASONING"],
        "vision": slot_default["VISION"], "qwen-vl": slot_default["VISION"],
        "vision-fast": slot_default["VISION_ALT"], "glm": slot_default["VISION_ALT"],
        "embed": slot_default["EMBED"], "nomic": slot_default["EMBED"],
        "general": slot_default["GENERAL"], "gemma": slot_default["GENERAL"],
        "long": slot_default["LONG"],
    })
    # Concrete configured models always resolve to themselves.
    if CHAT_MODEL:
        m.setdefault(CHAT_MODEL, CHAT_MODEL)
    if EMBED_MODEL:
        m.setdefault(EMBED_MODEL, EMBED_MODEL)
    return m


def _load_config() -> None:
    """(Re)read configuration from the environment. Called at import and by tests."""
    global UPSTREAMS, LOG_PATH, TIMEOUT, LOG_MAX_BYTES, PORT, HOST, CHAT_MODEL, EMBED_MODEL
    global MODEL_ALIASES, CB_FAILURE_THRESHOLD, CB_OPEN_SECONDS, MAX_RETRIES
    global RETRY_BASE_MS, PII_MASKING, QUOTA_TOKENS_PER_DAY, COST_PER_1M_USD
    global CLOUD_UPSTREAM, CLOUD_API_KEY, CLOUD_MODEL_REMAP, CLOUD_MODEL

    raw = os.environ.get("LLM_UPSTREAM", DEFAULT_UPSTREAM)
    UPSTREAMS = [u for u in (_normalise_upstream(part) for part in raw.split(",")) if u] or [
        _normalise_upstream(DEFAULT_UPSTREAM)
    ]
    LOG_PATH = Path(os.environ.get("LLM_LOG_PATH", str(DEFAULT_LOG_PATH)))
    TIMEOUT = float(os.environ.get("LLM_TIMEOUT", DEFAULT_TIMEOUT))
    LOG_MAX_BYTES = max(1, int(float(os.environ.get("LLM_LOG_MAX_MB", DEFAULT_LOG_MAX_MB)) * 1024 * 1024))
    # PORT is canonical; GATEWAY_PORT is a deprecated alias and only used if PORT is unset.
    PORT = int(os.environ.get("PORT") or os.environ.get("GATEWAY_PORT") or DEFAULT_PORT)
    HOST = os.environ.get("GATEWAY_HOST") or DEFAULT_HOST
    CHAT_MODEL = os.environ.get("LLM_CHAT_MODEL", "").strip()
    EMBED_MODEL = os.environ.get("LLM_EMBED_MODEL", "").strip()
    CB_FAILURE_THRESHOLD = max(1, int(os.environ.get("LLM_CB_FAILURE_THRESHOLD", "5")))
    CB_OPEN_SECONDS = max(1.0, float(os.environ.get("LLM_CB_OPEN_SECONDS", "30")))
    MAX_RETRIES = max(0, int(os.environ.get("LLM_MAX_RETRIES", "2")))
    RETRY_BASE_MS = max(0.0, float(os.environ.get("LLM_RETRY_BASE_MS", "50")))
    PII_MASKING = os.environ.get("LLM_PII_MASKING", "1").strip().lower() not in ("0", "false", "no", "off")
    QUOTA_TOKENS_PER_DAY = max(0, int(os.environ.get("LLM_QUOTA_TOKENS_PER_DAY", "0")))
    COST_PER_1M_USD = float(os.environ.get("LLM_COST_PER_1M_USD", "10.0"))
    _cloud_raw = os.environ.get("LLM_CLOUD_UPSTREAM", "").strip()
    CLOUD_UPSTREAM = _normalise_upstream(_cloud_raw) if _cloud_raw else ""
    CLOUD_API_KEY = os.environ.get("LLM_CLOUD_API_KEY", "").strip()
    CLOUD_MODEL = os.environ.get("LLM_CLOUD_MODEL", "").strip()
    CLOUD_MODEL_REMAP = os.environ.get("LLM_CLOUD_MODEL_REMAP", "1").strip().lower() not in ("0", "false", "no", "off")
    MODEL_ALIASES = _build_aliases()

    _reset_upstream_state()
    _reset_probe_cache()

    log = globals().get("telemetry")
    if log is not None:  # absent on the import-time call
        log.close()
        log.path = LOG_PATH
        log.max_bytes = LOG_MAX_BYTES


_load_config()

# ---------------------------------------------------------------- telemetry log


class TelemetryLog:
    """Append-only JSONL writer with size-based rotation keeping exactly 1 backup."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self._fh = None

    def _open(self):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        return self._fh

    def _rotate_if_needed(self) -> None:
        try:
            if not self.path.exists() or self.path.stat().st_size < self.max_bytes:
                return
        except OSError:
            return
        if self._fh is not None:
            with contextlib.suppress(Exception):
                self._fh.close()
            self._fh = None
        backup = self.path.with_name(self.path.name + ".1")
        with contextlib.suppress(OSError):
            backup.unlink()
        with contextlib.suppress(OSError):
            self.path.replace(backup)

    def write(self, record: dict) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            self._rotate_if_needed()
            fh = self._open()
            fh.write(line)
            fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(Exception):
                self._fh.close()
            self._fh = None


telemetry = TelemetryLog(LOG_PATH, LOG_MAX_BYTES)

# ---------------------------------------------------------------- upstream selection


def _candidates() -> list[str]:
    """Sticky selection: the last known-good upstream is tried first, then the rest."""
    order = list(UPSTREAMS)
    if _sticky_index is not None and 0 <= _sticky_index < len(order):
        order.insert(0, order.pop(_sticky_index))
    return order


def _mark_sticky(base: str) -> None:
    global _sticky_index
    if base in UPSTREAMS:
        _sticky_index = UPSTREAMS.index(base)


def _drop_sticky(base: str) -> None:
    global _sticky_index
    if _sticky_index is not None and 0 <= _sticky_index < len(UPSTREAMS) and UPSTREAMS[_sticky_index] == base:
        _sticky_index = None


async def _probe_one(client: httpx.AsyncClient, base: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.get(f"{base}/models", timeout=PROBE_TIMEOUT_S)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        if 200 <= r.status_code < 300:
            _mark_sticky(base)
            metrics_registry.set_gauge("llm_upstream_up", 1.0, {"upstream": base})
            return {"url": base, "reachable": True, "latency_ms": latency, "error": None}
        metrics_registry.set_gauge("llm_upstream_up", 0.0, {"upstream": base})
        return {"url": base, "reachable": False, "latency_ms": latency, "error": f"upstream_{r.status_code}"}
    except Exception as exc:  # noqa: BLE001 - a probe must never raise
        _drop_sticky(base)
        metrics_registry.set_gauge("llm_upstream_up", 0.0, {"upstream": base})
        return {
            "url": base,
            "reachable": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}"[:ERR_SNIPPET_CHARS],
        }


async def _probe(client: httpx.AsyncClient, force: bool = False) -> dict:
    """Cheap, TTL-cached (5s) reachability probe of the fallback chain."""
    now = time.monotonic()
    cached = _probe_cache.get("result")
    if not force and cached is not None and (now - _probe_cache["at"]) < PROBE_TTL_S:
        return dict(cached)

    last: dict = {"url": UPSTREAMS[0], "reachable": False, "latency_ms": 0.0, "error": "not_probed"}
    for base in _candidates():
        last = await _probe_one(client, base)
        if last["reachable"]:
            break
    _probe_cache["at"] = now
    _probe_cache["result"] = dict(last)
    return dict(last)


async def _resolve_upstream(client: httpx.AsyncClient) -> str:
    """Base URL to use for a request: the first reachable candidate, else the first one."""
    probe = await _probe(client)
    if probe.get("reachable"):
        return probe["url"]
    return _candidates()[0]

# ---------------------------------------------------------------- helpers


def resolve_model_alias(model: str, task: str | None = None, has_image: bool = False) -> str:
    """Map alias/task name to a concrete model id.

    Priority: explicit X-Task header > alias in `model` > vision auto-detect
    (image payload with empty/generic model) > configured default.
    Unknown concrete ids pass through untouched (BYO model).
    """
    m = (model or "").strip()
    t = (task or "").strip().lower()
    if t in MODEL_ALIASES:
        return MODEL_ALIASES[t]
    if has_image and (not m or m.lower() in ("fast", "reasoning", "general", "default", "auto")):
        return MODEL_ALIASES.get("vision", "qwen2.5-vl-3b-instruct")
    if m.lower() in MODEL_ALIASES:
        return MODEL_ALIASES[m.lower()]
    if m and m.lower() != "auto":
        return m
    return CHAT_MODEL or MODEL_ALIASES.get("general", "qwen2.5-vl-3b-instruct")


def _detect_image(messages: list) -> bool:
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and item.get("type") in ("image_url", "image"):
                        return True
            elif isinstance(c, str) and ("data:image" in c or "image_url" in c):
                return True
    return False


# ---------------------------------------------------------------- PII masking (VN)


def _mask_pii_in_text(text: str) -> tuple[str, int]:
    """Mask VN phone / CCCD / CMND in one string. Returns (masked, count)."""
    if not PII_MASKING or not isinstance(text, str) or not text:
        return text, 0
    total = 0
    masked, n1 = PII_PHONE_RE.subn(PII_TOKEN, text)
    total += n1
    masked, n2 = PII_CCCD_RE.subn(PII_TOKEN, masked)
    total += n2
    # CMND 9 digits: only mask when it looks standalone (avoid masking random ids).
    masked, n3 = PII_CMND_RE.subn(PII_TOKEN, masked)
    # Heuristic guard: a lone 9-digit run inside a long numeric string was already
    # handled by the CCCD rule; keep the count but do not over-claim on short texts.
    total += n3
    return masked, total


def _pii_mask_body(body: dict) -> tuple[dict, int]:
    """Return (masked_body, redactions). Never mutates the caller's dict."""
    if not PII_MASKING or not isinstance(body, dict):
        return body, 0
    import copy
    masked = copy.deepcopy(body)
    total = 0
    msgs = masked.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, str):
                m["content"], n = _mask_pii_in_text(c)
                total += n
            elif isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        item["text"], n = _mask_pii_in_text(item["text"])
                        total += n
    inp = masked.get("input")
    if isinstance(inp, str):
        masked["input"], n = _mask_pii_in_text(inp)
        total += n
    elif isinstance(inp, list):
        for i, x in enumerate(inp):
            if isinstance(x, str):
                inp[i], n = _mask_pii_in_text(x)
                total += n
    return masked, total


# ---------------------------------------------------------------- circuit breaker


def _cb_entry(base: str) -> dict[str, Any]:
    with _state_lock:
        entry = _cb_state.get(base)
        if entry is None:
            entry = {"failures": 0, "state": "closed", "opened_at": 0.0}
            _cb_state[base] = entry
        return entry


def _cb_allow(base: str) -> bool:
    """Closed -> allow; Open -> deny until OPEN_SECONDS elapse, then half-open trial."""
    with _state_lock:
        entry = _cb_state.get(base)
        if entry is None:
            return True
        if entry["state"] == "open":
            if (time.monotonic() - entry["opened_at"]) >= CB_OPEN_SECONDS:
                entry["state"] = "half-open"
                return True
            return False
        return True


def _cb_record_success(base: str) -> None:
    with _state_lock:
        entry = _cb_state.setdefault(base, {"failures": 0, "state": "closed", "opened_at": 0.0})
        entry["failures"] = 0
        if entry["state"] != "closed":
            entry["state"] = "closed"
    metrics_registry.set_gauge("llm_circuit_breaker_open", 0.0, {"upstream": base})


def _cb_record_failure(base: str) -> None:
    opened = False
    with _state_lock:
        entry = _cb_state.setdefault(base, {"failures": 0, "state": "closed", "opened_at": 0.0})
        entry["failures"] = int(entry.get("failures", 0)) + 1
        if entry["state"] == "half-open" or entry["failures"] >= CB_FAILURE_THRESHOLD:
            entry["state"] = "open"
            entry["opened_at"] = time.monotonic()
            opened = True
    if opened:
        metrics_registry.set_gauge("llm_circuit_breaker_open", 1.0, {"upstream": base})
        metrics_registry.inc("llm_circuit_breaker_opens_total", {"upstream": base})


def _retry_backoff_s(attempt: int) -> float:
    """Exponential backoff with full jitter: sleep ~ U(0, base * 2**attempt)."""
    cap_ms = RETRY_BASE_MS * (2 ** max(0, attempt))
    return random.uniform(0, cap_ms) / 1000.0


# ---------------------------------------------------------------- quota & cost


def _quota_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _estimate_tokens(prompt_chars: int, completion_chars: int = 0) -> int:
    return max(1, prompt_chars // 4) + max(0, completion_chars // 4)


def _quota_check(project: str, est_tokens: int) -> tuple[bool, int]:
    """Return (allowed, already_used). Unlimited when QUOTA_TOKENS_PER_DAY == 0."""
    if QUOTA_TOKENS_PER_DAY <= 0:
        return True, 0
    key = (project, _quota_day())
    with _state_lock:
        used = _quota_usage.get(key, 0)
    return (used + est_tokens <= QUOTA_TOKENS_PER_DAY), used


def _quota_add(project: str, tokens: int) -> int:
    key = (project, _quota_day())
    with _state_lock:
        _quota_usage[key] = _quota_usage.get(key, 0) + max(0, int(tokens))
        total = _quota_usage[key]
    metrics_registry.set_gauge("llm_quota_usage_tokens", float(total), {"project": project})
    return total


def _json_or_none(r: httpx.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return None


def _snippet(text: str) -> str:
    return " ".join((text or "").split())[:ERR_SNIPPET_CHARS]


def _prompt_stats(body: dict) -> tuple[int, str]:
    """Return (char_count, sha256[:16]) of the prompt. The prompt body is never logged."""
    try:
        if isinstance(body.get("messages"), list):
            payload = body["messages"]
            chars = sum(len(str(m.get("content", ""))) for m in payload if isinstance(m, dict))
        else:
            payload = body.get("input", "")
            if isinstance(payload, list):
                chars = sum(len(str(x)) for x in payload)
            else:
                chars = len(str(payload or ""))
    except Exception:
        chars, payload = -1, []
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
    return chars, digest


def _completion_chars(path: str, data: Any) -> int:
    if not isinstance(data, dict):
        return 0
    try:
        if path == "chat/completions":
            return len(str((data.get("choices") or [{}])[0].get("message", {}).get("content", "")))
        if path == "embeddings":
            return len(str((data.get("data") or [{}])[0].get("embedding", [])))
    except Exception:
        pass
    return 0


def _json_response(payload: dict, status_code: int) -> Response:
    return Response(content=json.dumps(payload), status_code=status_code, media_type="application/json")

# ---------------------------------------------------------------- app


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=TIMEOUT)
    telemetry._rotate_if_needed()
    telemetry._open()
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            await app.state.client.aclose()
        telemetry.close()


app = FastAPI(title="llm-gateway", version="2.0.0", lifespan=lifespan)


def _client(request_app: FastAPI) -> httpx.AsyncClient:
    client = getattr(request_app.state, "client", None)
    if client is None:  # only reachable if the lifespan did not run
        client = httpx.AsyncClient(timeout=TIMEOUT)
        request_app.state.client = client
    return client


@app.get("/health")
async def health(request: Request):
    probe = await _probe(_client(request.app))
    return {
        "status": "ok" if probe["reachable"] else "degraded",
        "upstream": probe,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health/ready")
async def health_ready(request: Request):
    probe = await _probe(_client(request.app))
    ready = bool(probe["reachable"])
    return _json_response({"status": "ready" if ready else "not_ready", "upstream": probe}, 200 if ready else 503)


@app.get("/metrics")
async def metrics():
    """Prometheus text exposition endpoint."""
    return Response(content=metrics_registry.to_prometheus_text(), media_type="text/plain; version=0.0.4")


@app.get("/admin/stats")
async def admin_stats():
    """Aggregated statistics from the telemetry log."""
    if not LOG_PATH.exists():
        return {"projects": {}, "total_requests": 0, "status": "no_telemetry_yet"}

    records_by_project: dict[str, list[dict]] = {}
    total = 0
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    proj = rec.get("project", "unknown")
                    records_by_project.setdefault(proj, []).append(rec)
                    total += 1
                except Exception:
                    continue
    except Exception as exc:
        return {"error": f"Failed to parse log: {exc}"}

    summary: dict[str, Any] = {}
    for proj, recs in records_by_project.items():
        latencies = sorted(r.get("latency_ms", 0.0) for r in recs if "latency_ms" in r)
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        errors = sum(1 for r in recs if r.get("status", 0) >= 400 or r.get("error"))
        prompt_chars = sum(r.get("prompt_chars", 0) for r in recs if r.get("prompt_chars", 0) > 0)
        completion_chars = sum(r.get("completion_chars", 0) for r in recs if r.get("completion_chars", 0) > 0)
        est_tokens = (prompt_chars + completion_chars) // 4
        est_commercial_savings_usd = round((est_tokens / 1_000_000) * 10.0, 4)

        summary[proj] = {
            "requests": len(recs),
            "errors": errors,
            "error_rate": round(errors / len(recs), 3) if recs else 0.0,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "est_tokens": est_tokens,
            "est_commercial_cost_saved_usd": est_commercial_savings_usd,
            "models_used": list({r.get("model") for r in recs if r.get("model")}),
        }

    return {"total_requests": total, "projects": summary, "timestamp": datetime.now(timezone.utc).isoformat()}



@app.get("/v1/routing")
async def routing_table():
    """Alias -> concrete model map + routing policy (for interview evidence)."""
    return {
        "aliases": dict(MODEL_ALIASES),
        "policy": "X-Task header > model alias > vision auto-detect (image payload) > LLM_CHAT_MODEL default",
        "vision_auto_detect": True,
        "task_header": "X-Task: fast | reasoning | vision | vision-fast | embed | general | long | auto",
        "cloud_failover": bool(CLOUD_UPSTREAM),
        "cloud_model": CLOUD_MODEL or None,
    }


@app.get("/v1/quotas")
async def quotas():
    """Per-project token usage vs the daily quota (0 = unlimited)."""
    day = _quota_day()
    usage = {proj: toks for (proj, d), toks in _quota_usage.items() if d == day}
    return {
        "date": day,
        "quota_tokens_per_day": QUOTA_TOKENS_PER_DAY,
        "cost_per_1m_usd": COST_PER_1M_USD,
        "usage_tokens": usage,
    }


@app.get("/v1/models")
async def models(request: Request):
    client = _client(request.app)
    base = await _resolve_upstream(client)
    try:
        r = await client.get(f"{base}/models")
    except Exception as exc:  # noqa: BLE001
        return _json_response({"error": {"message": f"{type(exc).__name__}: {exc}"[:300]}}, 502)

    data = _json_or_none(r)
    if r.status_code == 200 and isinstance(data, dict) and isinstance(data.get("data"), list):
        present = {str(m.get("id")) for m in data["data"] if isinstance(m, dict)}
        for model_id in (CHAT_MODEL, EMBED_MODEL):
            if model_id and model_id not in present:
                data["data"].append(
                    {"id": model_id, "object": "model", "owned_by": "llm-gateway", "configured": True}
                )
        return _json_response(data, 200)

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


async def _send(client: httpx.AsyncClient, path: str, body: dict) -> tuple[httpx.Response, str]:
    """POST with per-upstream circuit breaker + retry (exp backoff, full jitter).

    - Transport errors and 429/502/503/504 are retried up to MAX_RETRIES per base.
    - Other 4xx/5xx are passed through unchanged (caller contract), but 5xx
      still counts toward the breaker so a burning upstream eventually opens.
    - An open breaker skips the base entirely (fail-fast, no hanging on dead LAN).
    """
    errors: list[str] = []
    for base in _candidates():
        if not _cb_allow(base):
            errors.append(f"{base}: circuit_open")
            continue
        attempt = 0
        while True:
            try:
                r = await client.post(f"{base}/{path}", json=body)
            except Exception as exc:  # noqa: BLE001 - fall through to the next candidate
                _cb_record_failure(base)
                _drop_sticky(base)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(_retry_backoff_s(attempt))
                    attempt += 1
                    continue
                errors.append(f"{base}: {type(exc).__name__}")
                break
            if r.status_code in RETRYABLE_STATUSES and attempt < MAX_RETRIES:
                _cb_record_failure(base)
                await asyncio.sleep(_retry_backoff_s(attempt))
                attempt += 1
                continue
            if 200 <= r.status_code < 300:
                _cb_record_success(base)
            elif r.status_code >= 500:
                _cb_record_failure(base)
            _mark_sticky(base)
            return r, base
    if CLOUD_UPSTREAM and _cb_allow(CLOUD_UPSTREAM):
        # Hybrid failover: LAN chain exhausted, try the metered cloud base once
        # (bounded: single attempt + at most one retry) so a demo never dies.
        cloud_body = dict(body)
        if CLOUD_MODEL_REMAP and CLOUD_MODEL and isinstance(cloud_body.get("model"), str):
            cloud_body["model"] = CLOUD_MODEL
        headers = {"Authorization": f"Bearer {CLOUD_API_KEY}"} if CLOUD_API_KEY else None
        for attempt in range(min(1, MAX_RETRIES) + 1):
            try:
                r = await client.post(f"{CLOUD_UPSTREAM}/{path}", json=cloud_body, headers=headers)
            except Exception as exc:  # noqa: BLE001
                _cb_record_failure(CLOUD_UPSTREAM)
                if attempt < min(1, MAX_RETRIES):
                    await asyncio.sleep(_retry_backoff_s(attempt))
                    continue
                errors.append(f"{CLOUD_UPSTREAM}: {type(exc).__name__}")
                break
            if r.status_code in RETRYABLE_STATUSES and attempt < min(1, MAX_RETRIES):
                _cb_record_failure(CLOUD_UPSTREAM)
                await asyncio.sleep(_retry_backoff_s(attempt))
                continue
            if 200 <= r.status_code < 300:
                _cb_record_success(CLOUD_UPSTREAM)
            elif r.status_code >= 500:
                _cb_record_failure(CLOUD_UPSTREAM)
            metrics_registry.inc("llm_cloud_requests_total", {"status": str(r.status_code)})
            return r, CLOUD_UPSTREAM
    raise RuntimeError("all upstreams failed: " + "; ".join(errors))


async def _forward(path: str, req: Request) -> Response:
    raw = await req.body()
    try:
        body = json.loads(raw) if raw else {}
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    project = req.headers.get("X-Project", "unknown")
    task = req.headers.get("X-Task", "") or (body.get("task", "") if isinstance(body.get("task"), str) else "")
    rid = uuid.uuid4().hex[:16]

    # Model alias & vision auto-routing (X-Task > alias > vision auto-detect).
    has_image = _detect_image(body.get("messages", [])) if isinstance(body.get("messages"), list) else False
    orig_model = body.get("model", "")
    target_model = resolve_model_alias(orig_model, task=task, has_image=has_image)
    if isinstance(body.get("model"), str) and orig_model != target_model:
        body["model"] = target_model
    elif path == "embeddings" and not body.get("model"):
        body["model"] = target_model

    # PII masking (VN phone / CCCD) BEFORE anything leaves the gateway.
    body, pii_redactions = _pii_mask_body(body)
    if pii_redactions:
        metrics_registry.inc("llm_pii_redactions_total", {"project": project}, by=float(pii_redactions))

    prompt_chars, prompt_hash = _prompt_stats(body)
    t0 = time.perf_counter()
    status, err, usage, completion_chars, upstream_used = 0, None, {}, 0, None
    is_stream = bool(body.get("stream", False))

    # Token quota gate (per project per UTC day; 0 = unlimited).
    est_input_tokens = _estimate_tokens(prompt_chars)
    allowed, quota_used = _quota_check(project, est_input_tokens)
    if not allowed:
        status, err = 429, "quota_exceeded"
        metrics_registry.inc("llm_quota_denied_total", {"project": project})
        metrics_registry.inc("llm_requests_total", {"project": project, "status": "429"})
        telemetry.write(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "project": project,
                "model": target_model,
                "requested_model": orig_model,
                "task": task,
                "endpoint": f"/v1/{path}",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "prompt_chars": prompt_chars,
                "prompt_sha256": prompt_hash,
                "completion_chars": 0,
                "usage": {},
                "status": 429,
                "error": f"quota_exceeded: {quota_used}/{QUOTA_TOKENS_PER_DAY} tokens/day",
                "id": rid,
                "upstream": None,
                "pii_redactions": pii_redactions,
            }
        )
        return _json_response(
            {"error": {"message": f"daily token quota exceeded ({quota_used}/{QUOTA_TOKENS_PER_DAY})", "type": "quota_exceeded"}},
            429,
        )

    client = _client(req.app)

    # Streaming passthrough
    if is_stream and path == "chat/completions":
        base = await _resolve_upstream(client)
        if not _cb_allow(base):
            for cand in _candidates():
                if _cb_allow(cand):
                    base = cand
                    break

        async def _stream_generator():
            nonlocal status, upstream_used, err
            upstream_used = base
            status = 200
            try:
                async with client.stream("POST", f"{base}/{path}", json=body) as upstream_resp:
                    status = upstream_resp.status_code
                    async for chunk in upstream_resp.aiter_raw():
                        yield chunk
            except Exception as stream_exc:
                err = f"stream_error: {stream_exc}"
                yield f"data: {json.dumps({'error': {'message': str(stream_exc)}})}\n\n".encode()
            finally:
                lat = round((time.perf_counter() - t0) * 1000, 1)
                metrics_registry.inc("llm_requests_total", {"project": project, "status": str(status)})
                metrics_registry.observe("llm_latency_seconds", lat / 1000.0, {"project": project})
                telemetry.write(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "project": project,
                        "model": target_model,
                        "requested_model": orig_model,
                        "task": task,
                        "endpoint": f"/v1/{path}",
                        "latency_ms": lat,
                        "prompt_chars": prompt_chars,
                        "prompt_sha256": prompt_hash,
                        "completion_chars": -1,
                        "usage": {},
                        "status": status,
                        "error": err,
                        "id": rid,
                        "upstream": upstream_used,
                        "stream": True,
                        "pii_redactions": pii_redactions,
                    }
                )
                if 200 <= status < 300:
                    _quota_add(project, est_input_tokens)

        return StreamingResponse(
            _stream_generator(),
            media_type="text/event-stream",
            headers={"X-Request-Id": rid},
        )

    # Unary request
    try:
        r, upstream_used = await _send(client, path, body)
        status = r.status_code
        data = _json_or_none(r)
        if isinstance(data, dict) and isinstance(data.get("usage"), dict):
            usage = data["usage"]
        completion_chars = _completion_chars(path, data)
        if not 200 <= status < 300:
            err = f"upstream_{status}"
            snippet = _snippet(r.text)
            if snippet:
                err = f"{err}: {snippet}"
        out = Response(
            content=r.content,
            status_code=status,
            media_type=r.headers.get("content-type", "application/json"),
            headers={"X-Request-Id": rid},
        )
    except Exception as exc:  # noqa: BLE001 - the gateway must never crash the caller
        status = 502
        err = f"{type(exc).__name__}: {exc}"[:300]
        out = _json_response({"error": {"message": err, "type": "gateway_error"}}, 502)
        out.headers["X-Request-Id"] = rid

    lat_ms = round((time.perf_counter() - t0) * 1000, 1)
    metrics_registry.inc("llm_requests_total", {"project": project, "status": str(status)})
    metrics_registry.observe("llm_latency_seconds", lat_ms / 1000.0, {"project": project})
    if usage:
        prompt_tok = usage.get("prompt_tokens", 0)
        compl_tok = usage.get("completion_tokens", 0)
        if prompt_tok:
            metrics_registry.inc("llm_tokens_total", {"project": project, "type": "prompt"}, by=prompt_tok)
        if compl_tok:
            metrics_registry.inc("llm_tokens_total", {"project": project, "type": "completion"}, by=compl_tok)
    total_tokens = 0
    if isinstance(usage, dict) and usage.get("total_tokens"):
        try:
            total_tokens = int(usage["total_tokens"])
        except (TypeError, ValueError):
            total_tokens = 0
    if not total_tokens:
        total_tokens = _estimate_tokens(prompt_chars, completion_chars)
    if 200 <= status < 300:
        _quota_add(project, total_tokens)
        metrics_registry.inc(
            "llm_cost_saved_usd_total", {"project": project},
            by=round((total_tokens / 1_000_000) * COST_PER_1M_USD, 6),
        )

    telemetry.write(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "project": project,
            "model": target_model,
            "requested_model": orig_model,
            "task": task,
            "endpoint": f"/v1/{path}",
            "latency_ms": lat_ms,
            "prompt_chars": prompt_chars,
            "prompt_sha256": prompt_hash,
            "completion_chars": completion_chars,
            "usage": usage,
            "status": status,
            "error": err,
            "id": rid,
            "upstream": upstream_used or (UPSTREAMS[0] if UPSTREAMS else None),
            "pii_redactions": pii_redactions,
        }
    )
    return out


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    return await _forward("chat/completions", req)


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    return await _forward("embeddings", req)


# ---------------------------------------------------------------- entrypoint


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
