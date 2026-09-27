"""Tamper-evident audit hash chain (append-only, stdlib only).

Zero imports from the repo (hashlib + json + os only) so the module stays
import-clean for edge/test environments without cv2/torch/numpy or pydantic.

Log format: one JSON object per line (JSONL). Each stored entry::

    {"index": int, "prev_hash": str, "record": dict, "entry_hash": str}

``entry_hash = sha256(canonical_json({"index", "prev_hash", "record"}))``
where canonical JSON is ``json.dumps(..., sort_keys=True,
separators=(",", ":"), ensure_ascii=False)`` encoded as UTF-8.
Genesis ``prev_hash`` is the literal ``"GENESIS"``.

Phase 8 integration (2-line call site, NOT wired yet — existing DB-backed
``AuditLog`` logger in ``src/backend/service.py`` is left untouched):
``record_telemetry`` / ``resolve_ticket`` / ``resume_line`` append sites::

    from ..agent.audit_chain import append as audit_chain_append
    audit_chain_append({"action": audit.action, "ticket_id": audit.ticket_id,
                        "operator_id": audit.operator_id}, "/var/log/apex/audit.chain.jsonl")
"""

import hashlib
import json
import os

GENESIS_HASH = "GENESIS"


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _hash_entry(index: int, prev_hash: str, record: dict) -> str:
    payload = {"index": index, "prev_hash": prev_hash, "record": record}
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _read_lines(log_path) -> list:
    with open(str(log_path), "r", encoding="utf-8") as fh:
        return fh.readlines()


def append(record: dict, log_path) -> dict:
    """Append ``record`` to the chain file, returning the stored entry."""
    if not isinstance(record, dict):
        raise TypeError("record must be a dict")
    log_path = str(log_path)
    index = 0
    prev_hash = GENESIS_HASH
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prev_hash = entry["entry_hash"]
                index = int(entry["index"]) + 1
    entry_hash = _hash_entry(index, prev_hash, record)
    entry = {"index": index, "prev_hash": prev_hash, "record": record, "entry_hash": entry_hash}
    # Ensure parent dir exists; append atomically-ish (single O_APPEND write).
    parent = os.path.dirname(os.path.abspath(log_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    return entry


def verify(log_path) -> tuple:
    """Re-walk the chain. Returns ``(ok: bool, bad_index: int | None)``.

    - Missing file or empty/whitespace-only log -> ``(True, None)``.
    - Any malformed line, index gap, ``prev_hash`` mismatch, or recomputed
      hash mismatch -> ``(False, index_of_bad_line)``.
    - Partial-line truncation (crash mid-write) is a malformed last line,
      so it is reported at that line's index. Note: removing whole trailing
      lines cleanly yields a shorter-but-valid prefix and verifies ``True``;
      that is inherent to hash chains without an external expected length.
    """
    log_path = str(log_path)
    if not os.path.exists(log_path):
        return (True, None)
    try:
        lines = _read_lines(log_path)
    except FileNotFoundError:
        return (True, None)
    # Filter nothing: blank lines are skipped but do not break indexing of entries.
    expected_index = 0
    prev_hash = GENESIS_HASH
    physical_line = -1
    saw_entry = False
    for raw in lines:
        physical_line += 1
        if raw.strip() == "":
            continue
        saw_entry = True
        try:
            entry = json.loads(raw)
            idx = int(entry["index"])
            ph = str(entry["prev_hash"])
            rec = entry["record"]
            eh = str(entry["entry_hash"])
            if not isinstance(rec, dict):
                return (False, idx if str(idx).isdigit() else expected_index)
        except Exception:
            # Malformed line (incl. 1-byte tamper breaking JSON, or partial
            # truncation): attribute it to the expected position.
            return (False, expected_index)
        if idx != expected_index:
            return (False, idx)
        if ph != prev_hash:
            return (False, idx)
        if _hash_entry(idx, ph, rec) != eh:
            return (False, idx)
        prev_hash = eh
        expected_index += 1
    if not saw_entry:
        return (True, None)
    return (True, None)
