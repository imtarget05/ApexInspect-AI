"""Live infrastructure guards for integration and hard test suites.

Ensures that tests requiring live LAN LLM, Ollama, Qdrant, or databases
are gracefully skipped when run in offline CI environments or when services are down.
Only runs live tests if LIVE_TESTS=1 is set in the environment.
"""
from __future__ import annotations

import os
import socket
import urllib.request
import pytest

LIVE_TESTS_ENABLED = os.environ.get("LIVE_TESTS", "").lower() in ("1", "true", "yes")


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, OSError):
        return False


def is_http_reachable(url: str, timeout: float = 1.5) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        # Retry with GET in case HEAD is not allowed
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except Exception:
            return False


def require_live_llm(base_url: str = "http://127.0.0.1:8787/health"):
    if not LIVE_TESTS_ENABLED:
        pytest.skip("Skipping live LLM test: LIVE_TESTS!=1")
    if not is_http_reachable(base_url):
        pytest.skip(f"Skipping live LLM test: {base_url} is unreachable")


def require_db(kind: str = "postgres"):
    if not LIVE_TESTS_ENABLED:
        pytest.skip(f"Skipping live DB test ({kind}): LIVE_TESTS!=1")
