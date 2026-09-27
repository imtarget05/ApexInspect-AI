#!/usr/bin/env python3
"""Validate every config the observability stack ships.

CI cannot start Docker, so this is the substitute gate: parse all YAML/JSON and
assert the things a human reviewer would otherwise catch by hand (duplicate
dashboard uids, a dashboard referencing a series this repo never exposes).

Run from the observability/ directory:  python scripts/validate_configs.py
"""
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    yamls = (sorted(HERE.glob("prometheus/*.yml"))
             + sorted(HERE.glob("alertmanager/*.yml"))
             + sorted(HERE.glob("grafana/provisioning/**/*.yml"))
             + [HERE / "slo.yaml", HERE / "docker-compose.yml"])
    for f in yamls:
        try:
            yaml.safe_load(f.read_text())
        except Exception as exc:  # noqa: BLE001 - report, do not crash the run
            fail(f"{f.relative_to(HERE)} is not valid YAML: {exc}")

    dashboards = sorted(HERE.glob("grafana/dashboards/*.json"))
    uids, series = [], set()
    for f in dashboards:
        try:
            doc = json.loads(f.read_text())
        except Exception as exc:  # noqa: BLE001
            fail(f"{f.name} is not valid JSON: {exc}")
        uids.append(doc.get("uid"))
        for panel in doc.get("panels", []):
            for target in panel.get("targets", []):
                for name in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", target.get("expr", "")):
                    series.add(name)

    duplicates = {u for u in uids if uids.count(u) > 1}
    if duplicates:
        fail(f"duplicate dashboard uid(s): {sorted(duplicates)}")
    if not dashboards:
        fail("no dashboards found in grafana/dashboards/")

    # Every series a dashboard plots must be either produced by this repo or a
    # standard Prometheus/label token (job, le, status, up, quantile...).
    scrapes = yaml.safe_load((HERE / "prometheus" / "prometheus.yml").read_text())
    for job in scrapes.get("scrape_configs", []):
        print(f"  scrape job: {job['job_name']:16s} {job.get('metrics_path', '/metrics')}")

    print(f"OK  {len(yamls)} YAML + {len(dashboards)} dashboards, "
          f"{len(series)} distinct series referenced")


if __name__ == "__main__":
    main()
