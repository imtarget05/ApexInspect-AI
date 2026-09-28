"""Retrieval-level evaluation for the BM25 SOP retriever.

Complements ``eval/harness.py`` (agent-level: decision/grounding/content/
hallucination) with index-level metrics: Recall@k, Precision@k, MRR and Hit@1
for k = 1, 2, 3. A drop here means the *index* regressed, not the LLM.

Usage::

    python -m eval.retrieval_eval          # BM25 over data/sops, text report
    python -m eval.retrieval_eval --json   # machine-readable, for CI

Public API (stable — do not break callers):

    evaluate_retriever(retriever, dataset, top_k=3) -> RetrievalReport
    RetrievalReport.to_dict() / .to_text()
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOPS = os.path.join(_HERE, "data", "sops")

KS = (1, 2, 3)

_CORE_RE = re.compile(r"SOP-SMT-\d+", re.IGNORECASE)


def _core(sop_id: Any) -> str:
    """Canonical core of an SOP id so short ('SOP-SMT-001') and full
    ('SOP-SMT-001-solder-bridge') forms match each other."""
    text = str(sop_id or "")
    match = _CORE_RE.search(text)
    return match.group(0).upper() if match else text.strip().upper()


def is_relevant(retrieved_id: Any, expected_id: Any) -> bool:
    """True when a retrieved id satisfies one expected ground-truth id."""
    return _core(retrieved_id) == _core(expected_id)


def reciprocal_rank(ranked_ids: Sequence[Any], relevant: Sequence[Any]) -> float:
    """1/rank of the first relevant hit (rank is 1-based); 0.0 if none."""
    relevant_cores = {_core(r) for r in relevant}
    for rank, rid in enumerate(ranked_ids, start=1):
        if _core(rid) in relevant_cores:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: Sequence[Any], relevant: Sequence[Any], k: int) -> float:
    """Fraction of ground-truth SOPs found in the top-k retrieved ids."""
    if not relevant:
        return 0.0
    relevant_cores = {_core(r) for r in relevant}
    hits = {_core(rid) for rid in list(ranked_ids)[:k]} & relevant_cores
    return len(hits) / len(relevant_cores)


def precision_at_k(ranked_ids: Sequence[Any], relevant: Sequence[Any], k: int) -> float:
    """Fraction of the top-k retrieved ids that are ground-truth SOPs."""
    if k <= 0:
        return 0.0
    relevant_cores = {_core(r) for r in relevant}
    hits = sum(1 for rid in list(ranked_ids)[:k] if _core(rid) in relevant_cores)
    return hits / k


def hit_at_1(ranked_ids: Sequence[Any], relevant: Sequence[Any]) -> float:
    """1.0 when the top-1 hit is relevant, else 0.0."""
    if not ranked_ids:
        return 0.0
    relevant_cores = {_core(r) for r in relevant}
    return 1.0 if _core(ranked_ids[0]) in relevant_cores else 0.0


@dataclass
class QueryResult:
    """Per-query outcome: ranked ids plus rank of first relevant hit."""

    query_id: str
    query: str
    relevant: List[str] = field(default_factory=list)
    ranked: List[str] = field(default_factory=list)
    first_relevant_rank: int = 0  # 1-based; 0 = no relevant hit retrieved

    @property
    def hit(self) -> bool:
        return self.first_relevant_rank > 0


@dataclass
class RetrievalReport:
    """Aggregate retrieval metrics. Macro-averaged over queries."""

    results: List[QueryResult] = field(default_factory=list)
    top_k: int = 3

    @property
    def total(self) -> int:
        return len(self.results)

    def _mean(self, fn) -> float:
        if not self.results:
            return 0.0
        return sum(fn(r) for r in self.results) / len(self.results)

    def recall_at(self, k: int) -> float:
        return self._mean(lambda r: recall_at_k(r.ranked, r.relevant, k))

    def precision_at(self, k: int) -> float:
        return self._mean(lambda r: precision_at_k(r.ranked, r.relevant, k))

    @property
    def mrr(self) -> float:
        return self._mean(lambda r: reciprocal_rank(r.ranked, r.relevant))

    @property
    def hit_at_1(self) -> float:
        return self._mean(lambda r: hit_at_1(r.ranked, r.relevant))

    def to_dict(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total": self.total,
            "top_k": self.top_k,
            "mrr": round(self.mrr, 4),
            "hit_at_1": round(self.hit_at_1, 4),
        }
        for k in KS:
            if k <= self.top_k:
                summary[f"recall@{k}"] = round(self.recall_at(k), 4)
                summary[f"precision@{k}"] = round(self.precision_at(k), 4)
        return {
            "summary": summary,
            "queries": [
                {
                    "id": r.query_id,
                    "relevant": r.relevant,
                    "ranked": r.ranked,
                    "first_relevant_rank": r.first_relevant_rank,
                }
                for r in self.results
            ],
        }

    def to_text(self) -> str:
        lines = [
            f"SOP retriever evaluation — {self.total} queries (top_k={self.top_k})",
            f"  Recall@1: {self.recall_at(1):.1%}   "
            f"Recall@2: {self.recall_at(2):.1%}   "
            f"Recall@3: {self.recall_at(3):.1%}",
            f"  Precision@1: {self.precision_at(1):.1%}   "
            f"Precision@3: {self.precision_at(3):.1%}   "
            f"MRR: {self.mrr:.3f}   Hit@1: {self.hit_at_1:.1%}",
        ]
        for r in self.results:
            mark = "HIT " if r.hit else "MISS"
            rank_info = f"rank={r.first_relevant_rank}" if r.hit else "no relevant hit"
            lines.append(f"  [{mark}] {r.query_id} ({rank_info}) -> {r.ranked}")
        return "\n".join(lines)


def evaluate_retriever(
    retriever: Any,
    dataset: Sequence[Dict[str, Any]] | None = None,
    top_k: int = 3,
) -> RetrievalReport:
    """Score a retriever over a retrieval dataset.

    ``retriever`` is any object exposing ``search(query, top_k=)`` returning a
    list of dicts with an ``sop_id`` key — the real ``SOPRetriever`` or a fake
    in unit tests. ``dataset`` defaults to ``RETRIEVAL_QUERIES``. Metrics are
    reported for k = 1, 2, 3 (capped at ``top_k``); retrieval itself always
    fetches ``top_k`` hits per query.
    """
    if dataset is None:
        from eval.retrieval_dataset import RETRIEVAL_QUERIES

        dataset = RETRIEVAL_QUERIES

    report = RetrievalReport(top_k=top_k)
    for item in dataset:
        try:
            hits = retriever.search(item["query"], top_k=top_k) or []
        except Exception:
            hits = []
        ranked = [str(h.get("sop_id", "")) for h in hits if isinstance(h, dict)]
        relevant = [str(s) for s in item.get("relevant", [])]
        rr = reciprocal_rank(ranked, relevant)
        first_rank = int(round(1.0 / rr)) if rr > 0 else 0
        report.results.append(
            QueryResult(
                query_id=str(item.get("id", "?")),
                query=str(item.get("query", "")),
                relevant=relevant,
                ranked=ranked,
                first_relevant_rank=first_rank,
            )
        )
    return report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate BM25 SOP retrieval (Recall@k / MRR).")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--sops-dir", default=_SOPS, help="SOP corpus directory")
    parser.add_argument("--top-k", type=int, default=3, help="hits retrieved per query")
    args = parser.parse_args(argv)

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from src.agent.rag import SOPRetriever

    retriever = SOPRetriever(sops_dir=args.sops_dir)
    report = evaluate_retriever(retriever, top_k=args.top_k)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
