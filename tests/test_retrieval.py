#!/usr/bin/env python3
"""
Retrieval evaluation test.
Runs every row in tests/retrieval_eval.csv against RetrievalPipeline
and asserts min_results and slug pattern presence.

Usage:
    python tests/test_retrieval.py
    python tests/test_retrieval.py --verbose
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.retrieval_pipeline import RetrievalPipeline


def load_eval_cases(csv_path: Path) -> List[Dict]:
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cases.append({
                "query":                  row["query"].strip().strip('"'),
                "expected_slug_contains": row.get("expected_slug_contains", "").strip(),
                "category":               row.get("category", "").strip() or None,
                "min_results":            int(row.get("min_results", 1)),
            })
    return cases


def run_eval(cases: List[Dict], pipeline: RetrievalPipeline,
             verbose: bool = False) -> Tuple[int, int]:
    passed = failed = 0
    for case in cases:
        results = pipeline.retrieve(
            query=case["query"],
            category=case["category"],
        )
        slugs = [r.payload.get("slug", "") for r in results if r.payload]

        if len(results) < case["min_results"]:
            print(
                f"  [FAIL] '{case['query']}'"
                f" \u2014 got {len(results)} results, expected >= {case['min_results']}"
            )
            failed += 1
            continue

        pattern = case["expected_slug_contains"]
        if pattern:
            found = any(re.search(pattern, slug, re.IGNORECASE) for slug in slugs)
            if not found:
                print(
                    f"  [FAIL] '{case['query']}'"
                    f" \u2014 no slug matching '{pattern}' in: {slugs[:5]}"
                )
                failed += 1
                continue

        if verbose:
            top = slugs[0] if slugs else "\u2014"
            print(f"  [OK]   '{case['query']}' \u2192 {len(results)} results, top: {top}")
        passed += 1

    return passed, failed


def run_evaluation(verbose: bool = False) -> bool:
    """
    Callable entry point for app/cli.py evaluate.
    Returns True if all cases pass, False otherwise.
    """
    eval_csv = ROOT / "tests" / "retrieval_eval.csv"
    if not eval_csv.exists():
        print(f"Error: {eval_csv} not found.")
        return False

    cases    = load_eval_cases(eval_csv)
    pipeline = RetrievalPipeline.from_config()

    print(f"\nRunning {len(cases)} retrieval eval cases\u2026")
    passed, failed = run_eval(cases, pipeline, verbose=verbose)

    print(f"\n{'─' * 40}")
    print(f"Passed: {passed}  |  Failed: {failed}  |  Total: {passed + failed}")
    if failed == 0:
        print("\u2713 All eval cases passed.")
    else:
        print(f"\u2717 {failed} case(s) failed.")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Run retrieval eval cases from CSV.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ok = run_evaluation(verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
