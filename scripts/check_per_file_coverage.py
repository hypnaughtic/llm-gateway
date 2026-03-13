"""Enforce per-file test coverage minimum (default 90%).

Reads a coverage JSON report and fails if any source file falls below
the threshold. This is stricter than project-wide coverage: every file
must individually meet the bar.

Usage:
    pytest -m unit --cov=src --cov-report=json --cov-report=term
    python scripts/check_per_file_coverage.py [--threshold 90]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 90
COVERAGE_JSON = Path("coverage.json")


def main() -> int:
    """Check per-file coverage against threshold."""
    parser = argparse.ArgumentParser(description="Per-file coverage gate")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum coverage %% per file (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=COVERAGE_JSON,
        help=f"Path to coverage.json (default: {COVERAGE_JSON})",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"ERROR: {args.report} not found. Run pytest with --cov-report=json first.")
        return 1

    data = json.loads(args.report.read_text())
    files = data.get("files", {})

    failures: list[tuple[str, float]] = []
    for filepath, info in sorted(files.items()):
        summary = info.get("summary", {})
        pct = summary.get("percent_covered", 0.0)
        if pct < args.threshold:
            failures.append((filepath, pct))

    if not failures:
        print(f"All {len(files)} files meet {args.threshold}% per-file coverage.")
        return 0

    print(f"FAILED: {len(failures)} file(s) below {args.threshold}% coverage:\n")
    for filepath, pct in failures:
        print(f"  {pct:5.1f}%  {filepath}")
    print(f"\nThreshold: {args.threshold}%")
    return 1


if __name__ == "__main__":
    sys.exit(main())
