"""Retrieval-level golden dataset for the BM25 SOP retriever.

Why this exists
---------------
``eval/dataset.py`` + ``eval/harness.py`` score the *agent* (decision /
grounding / content / hallucination) but never measure the *retriever* in
isolation: if ``SOPRetriever.search`` silently degrades (bad synonym, broken
tokenizer, missing SOP file), the agent-level eval blames the LLM instead of
the index. This dataset pins retrieval quality with queries written against
the real wording of the six manuals in ``data/sops/``.

Schema per query
----------------
``id``        stable identifier, referenced by the report.
``query``     operator-style query string (EN keywords, VI synonyms, mixed).
``relevant``  1-2 ground-truth SOP ids (full file stems, e.g.
              ``SOP-SMT-001-solder-bridge``). Two entries only when both
              manuals genuinely answer the query (shared root cause).
``notes``     why this query exists (which SOP section it targets).
"""
from __future__ import annotations

from typing import Any, Dict, List

SOP_001 = "SOP-SMT-001-solder-bridge"
SOP_002 = "SOP-SMT-002-missing-component"
SOP_003 = "SOP-SMT-003-line-halt-safety"
SOP_004 = "SOP-SMT-004-ipc610-solder-criteria"
SOP_005 = "SOP-SMT-005-reflow-profiling-drift"
SOP_006 = "SOP-SMT-006-pick-place-nozzle-maintenance"

RETRIEVAL_QUERIES: List[Dict[str, Any]] = [
    {
        "id": "solder-bridge-stencil-en",
        "query": "solder bridge short circuit stencil squeegee pressure excess paste",
        "relevant": [SOP_001],
        "notes": "SOP-001 section 2.1: stencil underside + squeegee pressure root causes.",
    },
    {
        "id": "solder-bridge-vi-synonym",
        "query": "hàn chập cầu thiếc stencil lau IPA dao gạt",
        "relevant": [SOP_001],
        "notes": "Vietnamese synonym case: 'hàn chập' must hit SOP-001 via BM25 + synonym expansion.",
    },
    {
        "id": "solder-bridge-reflow-peak",
        "query": "reflow peak temperature 260 pre-heat flux splattering solder bridge",
        "relevant": [SOP_001],
        "notes": "SOP-001 section 2.2: pre-heat ramp / peak temp. Pins 001 over 005 via 'solder bridge'.",
    },
    {
        "id": "missing-component-feeder-en",
        "query": "missing component tape feeder jam R102 vision alignment",
        "relevant": [SOP_002],
        "notes": "SOP-002 sections 2.2-2.3: feeder exhaustion/jam + vision alignment error.",
    },
    {
        "id": "missing-component-vi-synonym",
        "query": "thiếu linh kiện vòi hút feeder kẹt băng không nạp liệu",
        "relevant": [SOP_002],
        "notes": "Vietnamese synonym case: 'thiếu linh kiện' / 'vòi hút' must hit SOP-002.",
    },
    {
        "id": "nozzle-missing-shared",
        "query": "vacuum nozzle clogged pick place missing component low suction pressure",
        "relevant": [SOP_002, SOP_006],
        "notes": "Multi-relevant: SOP-002 (nozzle root cause of missing) and SOP-006 "
                 "(nozzle clogging maintenance) both genuinely answer it.",
    },
    {
        "id": "line-halt-approval-en",
        "query": "halt line approval MES supervisor human in the loop restart procedure",
        "relevant": [SOP_003],
        "notes": "SOP-003 sections 3.1-3.4: HITL approval + MES ticket + safe restart.",
    },
    {
        "id": "line-halt-vi",
        "query": "dừng chuyền phê duyệt quản đốc ca khởi động lại MES conveyor lò hàn chạy chậm",
        "relevant": [SOP_003],
        "notes": "Vietnamese case: halt approval wording straight from SOP-003.",
    },
    {
        "id": "ipc610-wetting-fillet-en",
        "query": "IPC-A-610 wetting angle fillet height Class 3 high reliability",
        "relevant": [SOP_004],
        "notes": "SOP-004 section 2.1/2.3: wetting angle <= 90 deg + fillet height criteria.",
    },
    {
        "id": "ipc610-clearance-vi",
        "query": "khoảng cách điện môi 0.13 mm solder bridge spur clearance defect",
        "relevant": [SOP_004],
        "notes": "SOP-004 section 2.2: minimum electrical clearance. Pins 004 over 001 via clearance terms.",
    },
    {
        "id": "reflow-drift-pwi-en",
        "query": "reflow profile drift PWI TAL SAC305 peak temperature 245",
        "relevant": [SOP_005],
        "notes": "SOP-005 sections 1-2: profile drift definition + PWI/TAL/SAC305 control limits.",
    },
    {
        "id": "reflow-drift-kic-vi",
        "query": "trôi nhiệt lò hàn KIC profiler thermocouple hiệu chuẩn bù nhiệt quạt đối lưu",
        "relevant": [SOP_005],
        "notes": "Vietnamese case: KIC profiler + thermocouple wording from SOP-005 section 3.",
    },
    {
        "id": "nozzle-maintenance-en",
        "query": "nozzle tip wear ultrasonic cleaner IPA tombstoning vision calibration glass plate",
        "relevant": [SOP_006],
        "notes": "SOP-006 sections 2-3: tip wear, ultrasonic IPA bath, vision calibration plate.",
    },
    {
        "id": "nozzle-maintenance-vi",
        "query": "vệ sinh đầu hút sóng siêu âm khí nén cồn IPA kẹt feeder AOI báo mất linh kiện",
        "relevant": [SOP_006],
        "notes": "Vietnamese case: ultrasonic cleaning + compressed-air drying from SOP-006.",
    },
]


def by_id(query_id: str) -> Dict[str, Any]:
    for item in RETRIEVAL_QUERIES:
        if item["id"] == query_id:
            return item
    raise KeyError(f"unknown retrieval query: {query_id}")
