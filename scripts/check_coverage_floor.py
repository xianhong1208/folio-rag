
"""Per-file coverage-floor gate (coverage's fail_under only checks the overall
total, with no per-file threshold).

Usage (generate coverage.json first, then check):
    uv run --no-sync pytest --cov=src --cov-report=json:reports/coverage.json
    uv run --no-sync python scripts/check_coverage_floor.py reports/coverage.json

Rule: every file included in the stats must have line coverage >= FLOOR (50%);
offenders are listed and the script exits non-zero. The omit list
(pyproject [tool.coverage.run]) governs which files are included in the stats;
this gate ensures an included file cannot be left under-tested — together they
force a choice: either test it properly or explicitly list an exemption reason,
with no middle ground.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FLOOR = 50.0


def main() -> int:
    report = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/coverage.json")
    if not report.is_file():
        print(f"❌ coverage json not found: {report} (run pytest --cov-report=json:{report} first)")
        return 2

    data = json.loads(report.read_text(encoding="utf-8"))
    offenders = []
    for path, info in sorted(data.get("files", {}).items()):
        summary = info.get("summary", {})
        # Empty files (e.g. __init__ with 0 statements) are treated as passing
        if summary.get("num_statements", 0) == 0:
            continue
        pct = summary.get("percent_covered", 0.0)
        if pct < FLOOR:
            offenders.append((pct, path, summary.get("num_statements", 0)))

    if offenders:
        print(f"❌ {len(offenders)} file(s) below the per-file floor of {FLOOR:.0f}%:")
        for pct, path, stmts in sorted(offenders):
            print(f"   {pct:5.1f}%  {path}  ({stmts} stmts)")
        print("   -> add tests, or (for heavy external deps) add to pyproject omit with a reason.")
        return 1

    total = data.get("totals", {}).get("percent_covered", 0.0)
    print(f"✅ Per-file coverage floor of {FLOOR:.0f}% passed everywhere (overall {total:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
