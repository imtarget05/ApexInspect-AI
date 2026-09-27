# llm-gateway

An OpenAI-compatible HTTP proxy in front of the LAN LLM (LM Studio on `192.168.1.8:1234`,
with local Ollama as a fallback), plus **one centralized telemetry stream** for this
project's own AI calls. Runtime dependencies: `fastapi`, `uvicorn`, `httpx`. No auth —
it binds to loopback by default and is intended for a trusted LAN/developer machine.

> **Why this folder lives inside the project.** The gateway used to sit in a
> shared top-level folder, so cloning a single project gave you an app that could
> not reach any LLM. It is vendored here on purpose: `git clone <this repo>` is
> enough to run the whole stack — app + gateway + observability. The sibling
> `observability/` directory scrapes *this* gateway.

- Base URL for every client: `http://localhost:8787/v1`
- Point any OpenAI SDK at it: `OpenAI(base_url="http://localhost:8787/v1", api_key="not-used")`
- Only OpenAI shapes are exposed. Ollama-native `/api/*` is **not** proxied; use
  `/v1/chat/completions` and `/v1/embeddings`.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness, always 200 while the process is alive |
| `GET /health/ready` | 200 / 503 based on an upstream reachability probe |
| `GET /metrics` | Prometheus text exposition (`llm_requests_total`, `llm_latency_seconds`, …) |
| `GET /admin/stats` | per-project p50/p95, error rate, token + cost-saved rollup |
| `GET /v1/models` | upstream model list plus the configured chat/embed ids |
| `POST /v1/chat/completions` | chat + vision (SSE passthrough when `stream: true`) |
| `POST /v1/embeddings` | embeddings |

Send `X-Project: <project-name>` so telemetry is attributed per caller.

## Models served by the LAN upstream

7 models behind one port (`http://192.168.1.8:1234/v1`); pick one per task:

| Alias | Model | Use for |
|---|---|---|
| `reasoning` | `deepseek/deepseek-r1-0528-qwen3-8b` | complex reasoning, credit/agent decisions |
| `fast` | `nvidia/nemotron-3-nano-4b` | low-latency classification, guardrails |
| `general` | `google/gemma-4-e4b` | conversational / ITSM answers |
| `vision` | `qwen2.5-vl-3b-instruct` | PCB inspection, document understanding |
| `vision-fast` | `zai-org/glm-4.6v-flash` | fast vision alternative |
| `embed` | `text-embedding-nomic-embed-text-v1.5` | embeddings (RAG) |

A payload containing `image_url` parts is auto-routed to a vision model.


## Quickstart

Local venv:

```bash
cd llm-gateway
uv venv --python 3.12 .venv          # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env                  # then edit LLM_UPSTREAM

set -a; . ./.env; set +a              # export .env into the process environment
./.venv/bin/python server.py          # honours PORT, then the deprecated GATEWAY_PORT
```

`server.py` reads the **process environment**, not the `.env` file — there is no dotenv
dependency. Load the file into the environment as above, or run under uvicorn with
`uvicorn server:app --port 8787 --env-file .env` (works because `uvicorn[standard]`
bundles python-dotenv).

Docker:

```bash
cd llm-gateway
cp .env.example .env
docker compose up --build
docker compose ps                      # healthy once the upstream answers /health/ready
```

Tests:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest             # 39 tests, no network, no LAN
```

## v2: AI Control Plane (multi-model gateway)

| Capability | How | Evidence |
| --- | --- | --- |
| Task routing (7 LAN models) | `model: fast\|reasoning\|vision\|vision-fast\|embed\|general\|long` or `X-Task` header; image payloads auto-route to vision | `GET /v1/routing`, `tests: test_xtask_header_overrides_model_alias` |
| Circuit breaker | Closed → Open → Half-Open per upstream (`LLM_CB_FAILURE_THRESHOLD`, `LLM_CB_OPEN_SECONDS`); open skips fail-fast | `llm_circuit_breaker_open`, `test_circuit_breaker_opens_and_failfast` |
| Retry | Exp backoff + full jitter on transport errors and 429/502/503/504 (`LLM_MAX_RETRIES`, `LLM_RETRY_BASE_MS`) | `test_retry_recovers_from_transient_503` |
| SSE streaming | `{"stream": true}` proxied as `text/event-stream` instead of buffered | `test_streaming_passthrough_path` |
| PII masking (VN) | Phone (`0[35789]…`/`+84…`) + CCCD 12 số + CMND 9 số → `[REDACTED_PII]` trước khi gửi upstream | `llm_pii_redactions_total`, `test_pii_masking_redacts_vn_phone_and_cccd` |
| Quota & cost | Daily token quota/project (`LLM_QUOTA_TOKENS_PER_DAY`, 429 khi vượt) + `llm_cost_saved_usd_total` vs giá thương mại | `GET /v1/quotas`, `test_quota_denied_when_exceeded` |
| Metrics | `/metrics` Prometheus-format, zero extra deps (`metrics.py`) | `test_metrics_endpoint_returns_prometheus_format` |

Per-slot model override: `LLM_MODEL_FAST / _REASONING / _VISION / _VISION_ALT / _EMBED / _GENERAL / _LONG`.

## Configuration

All variables are read from the process environment (see `.env.example` for the full
commented table). `docker-compose.yml` loads them with `env_file: .env`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_UPSTREAM` | `http://192.168.1.8:1234/v1` | Comma-separated **fallback chain** of OpenAI-compatible bases. The first reachable entry wins and is then reused (sticky). A missing scheme becomes `http://`, a missing `/v1` is appended, a trailing `/api` is rewritten to `/v1`. |
| `LLM_CHAT_MODEL` | *(unset)* | Chat model id. Always listed by `GET /v1/models`, even if the upstream omits it. |
| `LLM_EMBED_MODEL` | *(unset)* | Embedding model id. Same treatment. |
| `LLM_TIMEOUT` | `120` | Per-request upstream timeout, seconds. |
| `LLM_LOG_PATH` | `./logs/llm-telemetry.jsonl` | Telemetry JSONL file. |
| `LLM_LOG_MAX_MB` | `10` | When the log exceeds this size it is rotated to `LLM_LOG_PATH.1`. Exactly 1 previous file is kept. |
| `PORT` | `8787` | **Canonical** listen port. |
| `GATEWAY_PORT` | *(unset)* | **Deprecated alias** for `PORT`; only read when `PORT` is unset. |
| `GATEWAY_HOST` | `127.0.0.1` | Bind address. Docker overrides it to `0.0.0.0` so the published port is reachable. |

Example chain — callers work whether LM Studio or Ollama is up:

```
LLM_UPSTREAM=http://192.168.1.8:1234/v1,http://127.0.0.1:11434/v1
```

## Endpoints

| Method | Path | Success | Notes |
| --- | --- | --- | --- |
| `GET` | `/health` | always `200` | Liveness. `status` is `ok` or `degraded`; HTTP never drops below 200 while the process is alive. |
| `GET` | `/health/ready` | `200` / `503` | Readiness. `503` means no upstream in the chain is reachable. Used by the compose healthcheck. |
| `GET` | `/v1/models` | upstream status | Upstream model list, with `LLM_CHAT_MODEL` / `LLM_EMBED_MODEL` added if missing. |
| `POST` | `/v1/chat/completions` | upstream status | OpenAI chat shape, passed through byte-for-byte. |
| `POST` | `/v1/embeddings` | upstream status | OpenAI embeddings shape; `input` may be a string or a list. |

`GET /health` body:

```json
{
  "status": "degraded",
  "upstream": {"url": "http://192.168.1.8:1234/v1", "reachable": false,
               "latency_ms": 2.7, "error": "ConnectError: connection refused"},
  "time": "2026-09-27T09:30:00.000000+00:00"
}
```

`GET /health/ready` returns `{"status": "ready"|"not_ready", "upstream": {...}}` with
HTTP `200` or `503` respectively.

### Request header

| Header | Values |
| --- | --- |
| `X-Project` | `MAIA`, `CreditFlow`, `Factory-Data-Automation`, `ApexInspect-AI`, `IT-Helpdesk-Lab`, or `unknown` (default when absent) |

If no upstream can be reached, `POST /v1/*` returns `502` with
`{"error": {"message": "...", "type": "gateway_error"}}`. Upstream non-2xx responses are
returned to the caller **unchanged** (same status, same body).

## Telemetry

One JSONL line per forwarded request, appended to `LLM_LOG_PATH`. **Prompt and completion
bodies are never stored** — only sizes and a truncated hash.

| Field | Meaning |
| --- | --- |
| `ts` | ISO-8601 UTC timestamp |
| `project` | `X-Project` header, or `unknown` |
| `model` | requested model id |
| `endpoint` | `/v1/chat/completions` or `/v1/embeddings` |
| `latency_ms` | wall time of the upstream call |
| `prompt_chars` | characters in the prompt (`messages[].content`, or `input`) |
| `prompt_sha256` | first 16 hex chars of sha256 of the prompt payload |
| `completion_chars` | chars of the assistant content (embeddings: `len(str(embedding))`) |
| `usage` | upstream `usage` object, or `{}` |
| `status` | HTTP status returned to the caller (200, or the upstream status, or 502) |
| `error` | `null`, `upstream_<status>: <body snippet>`, or the transport error |
| `id` | 16-hex request id (added in 1.1) |
| `upstream` | base URL that served the request (added in 1.1) |

Query it with `jq`:

```bash
# requests per project
jq -r '.project' logs/llm-telemetry.jsonl | sort | uniq -c | sort -rn

# failure rate per project (upstream errors are recorded, not silently passed through)
jq -r 'select(.error != null) | "\(.project)\t\(.status)\t\(.error)"' logs/llm-telemetry.jsonl

# average and p95 latency per endpoint
jq -s 'group_by(.endpoint) | map({endpoint: .[0].endpoint,
      avg_ms: (map(.latency_ms)|add/length),
      p95_ms: (map(.latency_ms)|sort|.[(length*0.95|floor)])})' logs/llm-telemetry.jsonl

# token totals per model
jq -s 'group_by(.model) | map({model: .[0].model, total_tokens: (map(.usage.total_tokens // 0)|add)})' \
  logs/llm-telemetry.jsonl

# slowest 10 requests
jq -s 'sort_by(-.latency_ms)[:10] | map({ts, project, endpoint, latency_ms})' logs/llm-telemetry.jsonl
```

Rotation: when `llm-telemetry.jsonl` reaches `LLM_LOG_MAX_MB`, it is renamed to
`llm-telemetry.jsonl.1` (replacing any previous `.1`) and a fresh file is started, so at
most 2 files exist.

## Clients wired to this gateway

| Project | Repo | `X-Project` value | Uses |
| --- | --- | --- | --- |
| MAIA | `MAIA/` | `MAIA` | `/v1/chat/completions` |
| CreditFlow | `CreditFlow/` | `CreditFlow` | `/v1/chat/completions` |
| Factory Data Automation + AI Reporting | `Factory-Data-Automation-v-AI-Reporting-Platform/` | `Factory-Data-Automation` | `/v1/chat/completions` |
| ApexInspect-AI | `ApexInspect-AI/` | `ApexInspect-AI` | `/v1/chat/completions`, `/v1/embeddings` |
| Enterprise IT Helpdesk Lab | `05-Enterprise-IT-Helpdesk-Lab/` | `IT-Helpdesk-Lab` | `/v1/chat/completions`, `/v1/embeddings` |
| FlashSale Backend | `FlashSale-Backend/` | *(not yet wired)* | planned |
| MiniERP Manufacturing / Warehouse | `04-MiniERP-Manufacturing-Warehouse/` | *(not yet wired)* | planned |

## Layout

```
server.py                 gateway (config, upstream chain, proxy, telemetry, log rotation)
tests/test_gateway.py     28 tests against an in-process fake upstream
tests/conftest.py         guard that fails the suite if a test touches the real log
.env / .env.example       configuration
logs/llm-telemetry.jsonl  telemetry (rotated to .1 at LLM_LOG_MAX_MB)
Dockerfile, docker-compose.yml
```
