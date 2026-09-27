"""Prometheus-compatible in-memory metric store without external dependencies."""
from __future__ import annotations

import threading

_LATENCY_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._hist_buckets: dict[str, dict[float, int]] = {}
        self._hist_sum: dict[str, float] = {}
        self._hist_count: dict[str, int] = {}

    def inc(self, name: str, labels: dict[str, str] | None = None, by: float = 1.0) -> None:
        key = self._format_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + by

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._format_key(name, labels)
        with self._lock:
            self._gauges[key] = float(value)

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._format_key(name, labels)
        with self._lock:
            if key not in self._hist_buckets:
                self._hist_buckets[key] = {b: 0 for b in _LATENCY_BUCKETS}
                self._hist_sum[key] = 0.0
                self._hist_count[key] = 0
            for b in _LATENCY_BUCKETS:
                if value <= b:
                    self._hist_buckets[key][b] += 1
            self._hist_sum[key] += value
            self._hist_count[key] += 1

    def _format_key(self, name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        lbls = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{lbls}}}"

    def to_prometheus_text(self) -> str:
        lines: list[str] = []
        with self._lock:
            for key, val in sorted(self._counters.items()):
                lines.append(f"{key} {val}")
            for key, val in sorted(self._gauges.items()):
                lines.append(f"{key} {val}")
            for key, buckets in sorted(self._hist_buckets.items()):
                base_name = key.split("{")[0]
                has_labels = "{" in key
                label_body = key.split("{")[1].rstrip("}") if has_labels else ""
                cumulative = 0
                for b in _LATENCY_BUCKETS:
                    cumulative += buckets[b]
                    le_lbl = f'le="{b}"'
                    lbl_str = f"{{{label_body},{le_lbl}}}" if label_body else f"{{{le_lbl}}}"
                    lines.append(f"{base_name}_bucket{lbl_str} {cumulative}")
                inf_lbl = 'le="+Inf"'
                lbl_str_inf = f"{{{label_body},{inf_lbl}}}" if label_body else f"{{{inf_lbl}}}"
                lines.append(f"{base_name}_bucket{lbl_str_inf} {self._hist_count[key]}")
                lbl_body_clean = f"{{{label_body}}}" if label_body else ""
                lines.append(f"{base_name}_sum{lbl_body_clean} {self._hist_sum[key]}")
                lines.append(f"{base_name}_count{lbl_body_clean} {self._hist_count[key]}")
        return "\n".join(lines) + "\n"


metrics_registry = MetricsRegistry()
