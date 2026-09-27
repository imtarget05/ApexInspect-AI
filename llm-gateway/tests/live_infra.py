"""Offline-first guard for live/infra tests.

Stdlib only (urllib) — deliberately no httpx/fastapi imports so that CI
runners without gateway dependencies can still collect this module.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest


def probe_upstream(url: str, timeout_s: float = 1.5) -> bool:
    """Return True iff ``url`` answers with a 2xx status within ``timeout_s``.

    Never raises: any transport error, timeout, or non-2xx status is False.
    """
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False
    except Exception:  # noqa: BLE001 - a probe must never raise
        return False


def _gateway_base() -> str:
    return os.environ.get("LLM_GATEWAY_URL", "http://127.0.0.1:8787").rstrip("/") or "http://127.0.0.1:8787"


def _lan_upstream_models_url() -> str:
    raw = os.environ.get("LLM_UPSTREAM", "http://192.168.1.8:1234/v1")
    first = (raw.split(",")[0] if raw else "").strip().rstrip("/") or "http://192.168.1.8:1234/v1"
    if not first.startswith(("http://", "https://")):
        first = "http://" + first
    if first.endswith("/models"):
        return first
    if first.endswith("/v1"):
        return first + "/models"
    return first + "/v1/models"


def require_live_llm() -> None:
    """Skip the calling test unless live infrastructure is reachable.

    Skips (pytest.skip) if the gateway ``/health/ready`` (falling back to
    ``/health``) or the LAN upstream ``/v1/models`` is unreachable within
    1.5s. Returns None when everything is reachable.
    """
    gateway = _gateway_base()
    if not probe_upstream(f"{gateway}/health/ready", timeout_s=1.5):
        if not probe_upstream(f"{gateway}/health", timeout_s=1.5):
            pytest.skip(f"live gateway unreachable at {gateway} (offline-first CI)")
            return
    upstream_models = _lan_upstream_models_url()
    if not probe_upstream(upstream_models, timeout_s=1.5):
        pytest.skip(f"live LAN upstream unreachable at {upstream_models} (offline-first CI)")
