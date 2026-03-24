#!/usr/bin/env python3
"""Evaluate FAISS retrieval quality with Hit Rate@k and MRR@k."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import config
from tools import vector_store

logger = logging.getLogger(__name__)


@dataclass
class EvalRow:
    query: str
    gold_sources: set[str]


def _normalize_source(value: str) -> str:
    """Normalize source path/url to file-name-like key."""
    return Path(value).name.strip().lower()


def _load_rows(path: Path) -> list[EvalRow]:
    """
    Load evaluation rows from JSONL.

    Expected schema per line:
    {"query":"...", "gold_sources":["fileA.pdf","fileB.pdf"]}
    """
    if not path.exists():
        raise FileNotFoundError(f"RAG eval dataset not found: {path}")

    out: list[EvalRow] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        row = json.loads(s)
        query = str(row.get("query", "")).strip()
        gold = row.get("gold_sources") or []
        if not query:
            raise ValueError(f"Line {i}: missing query")
        if not isinstance(gold, list) or not gold:
            raise ValueError(f"Line {i}: gold_sources must be non-empty list")
        out.append(
            EvalRow(
                query=query,
                gold_sources={_normalize_source(str(v)) for v in gold if str(v).strip()},
            )
        )
    if not out:
        raise ValueError(f"No valid eval rows in {path}")
    return out


def _first_relevant_rank(retrieved_sources: Iterable[str], gold_sources: set[str]) -> int | None:
    for rank, src in enumerate(retrieved_sources, start=1):
        if _normalize_source(src) in gold_sources:
            return rank
    return None


def evaluate_hit_rate_mrr(rows: list[EvalRow], k: int, *, index_path: str | None = None) -> dict:
    """Compute Hit Rate@k and MRR@k for FAISS retrieval."""
    if k <= 0:
        raise ValueError("k must be >= 1")

    hits = 0
    rr_sum = 0.0
    details: list[dict] = []

    for row in rows:
        docs = vector_store.search(row.query, k=k, path=index_path)
        retrieved = [str((d.metadata or {}).get("source", "")) for d in docs]
        first_rank = _first_relevant_rank(retrieved, row.gold_sources)

        is_hit = first_rank is not None
        if is_hit:
            hits += 1
            rr_sum += 1.0 / first_rank

        details.append(
            {
                "query": row.query,
                "hit": is_hit,
                "first_relevant_rank": first_rank,
                "retrieved_sources": retrieved,
                "gold_sources": sorted(row.gold_sources),
            }
        )

    n = len(rows)
    metrics = {
        "num_queries": n,
        "k": k,
        "hit_rate_at_k": hits / n,
        "mrr_at_k": rr_sum / n,
    }
    return {"metrics": metrics, "details": details}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate FAISS RAG with Hit Rate@k and MRR@k.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.RAG_EVAL_DATASET_PATH,
        help="JSONL dataset path (default: data/eval/rag_eval.jsonl).",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics (default: 5).")
    parser.add_argument(
        "--index-path",
        type=str,
        default=str(config.FAISS_INDEX_PATH),
        help="FAISS index directory path (default: config.FAISS_INDEX_PATH).",
    )
    parser.add_argument(
        "--save-details",
        type=Path,
        default=None,
        help="Optional JSON path to save per-query details.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.dataset)
    result = evaluate_hit_rate_mrr(rows, args.k, index_path=args.index_path)
    m = result["metrics"]
    logger.info("RAG Eval: queries=%s, k=%s", m["num_queries"], m["k"])
    logger.info("Hit Rate@%s = %.4f", m["k"], m["hit_rate_at_k"])
    logger.info("MRR@%s = %.4f", m["k"], m["mrr_at_k"])

    if args.save_details:
        args.save_details.parent.mkdir(parents=True, exist_ok=True)
        args.save_details.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved detailed eval result to %s", args.save_details)


if __name__ == "__main__":
    main()

