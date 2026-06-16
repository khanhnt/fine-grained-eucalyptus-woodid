#!/usr/bin/env python3
"""Compatibility wrapper for printing Split A/Split B pHash audit counts.

The maintained implementation lives in scripts/export_phash_audit_reports.py.
This wrapper keeps the original manifest-folder entry point usable.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    if "--summary-only" not in sys.argv:
        sys.argv.append("--summary-only")
    repo_root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(repo_root / "scripts" / "export_phash_audit_reports.py"), run_name="__main__")
