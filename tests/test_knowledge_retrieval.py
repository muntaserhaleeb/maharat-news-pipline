#!/usr/bin/env python3
"""
Knowledge retrieval evaluation.
Tests queries against the maharat_knowledge_live collection and
checks that results are non-empty and textually relevant.

Usage:
    python tests/test_knowledge_retrieval.py
    python tests/test_knowledge_retrieval.py --verbose
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.config_service import load_qdrant_config
from services.retrieval_service import RetrievalService


def _ok(label: str, detail: str = "") -> None:
    msg = f"  [OK]   {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)


def _fail(label: str, detail: str = "") -> None:
    msg = f"  [FAIL] {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)


def load_eval_cases(csv_path: Path) -> List[Dict]:
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cases.append({
                "query":                 row["query"].strip().strip('"'),
                "expected_text_contains": row.get("expected_text_contains", "").strip().lower(),
                "knowledge_type":        row.get("knowledge_type", "").strip() or None,
                "min_results":           int(row.get("min_results", 1)),
            })
    return cases


def run_knowledge_eval(
    cases: List[Dict],
    service: RetrievalService,
    verbose: bool = False,
) -> Tuple[int, int]:
    passed = failed = 0

    for case in cases:
        query   = case["query"]
        pattern = case["expected_text_contains"]
        k_type  = case["knowledge_type"]

        query_filter = service.build_filter(knowledge_type=k_type) if k_type else None

        results = service.search(
            query_text=query,
            limit=5,
            query_filter=query_filter,
        )

        if len(results) < case["min_results"]:
            _fail(f"'{query}'", f"got {len(results)} results, expected >= {case['min_results']}")
            failed += 1
            continue

        if pattern:
            combined = " ".join(
                (r.payload.get("chunk_text") or "") + " " +
                (r.payload.get("title") or "")
                for r in results
            ).lower()
            if pattern not in combined:
                _fail(
                    f"'{query}'",
                    f"expected '{pattern}' not found in top {len(results)} results",
                )
                failed += 1
                continue

        if verbose:
            top_title = (results[0].payload or {}).get("title", "—") if results else "—"
            top_score = results[0].score if results else 0.0
            top_ktype = (results[0].payload or {}).get("knowledge_type", "—") if results else "—"
            _ok(
                f"'{query}'",
                f"{len(results)} results  top: {top_title!r} [{top_ktype}] score={top_score:.4f}",
            )
        passed += 1

    return passed, failed


def run_knowledge_evaluation(verbose: bool = False) -> bool:
    """Callable entry point for app/cli.py evaluate-knowledge."""
    eval_csv = ROOT / "tests" / "knowledge_eval.csv"
    if not eval_csv.exists():
        print(f"Error: {eval_csv} not found.")
        return False

    qdrant_cfg = load_qdrant_config()
    try:
        service = RetrievalService.from_config(qdrant_cfg, collection_key="knowledge")
    except Exception as exc:
        print(f"Error: could not connect to knowledge collection — {exc}")
        print("Run: python app/cli.py ingest-knowledge")
        return False

    cases = load_eval_cases(eval_csv)
    print(f"\nRunning {len(cases)} knowledge retrieval eval cases…")
    passed, failed = run_knowledge_eval(cases, service, verbose=verbose)

    print(f"\n{'─' * 40}")
    print(f"Passed: {passed}  |  Failed: {failed}  |  Total: {passed + failed}")
    if failed == 0:
        print("✓ All knowledge eval cases passed.")
    else:
        print(f"✗ {failed} case(s) failed.")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run knowledge retrieval eval cases."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ok = run_knowledge_evaluation(verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
