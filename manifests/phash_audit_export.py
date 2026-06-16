#!/usr/bin/env python3
"""Compatibility wrapper for exporting pHash audit CSV reports.

The maintained implementation lives in scripts/export_phash_audit_reports.py.
This wrapper keeps the original manifest-folder entry point usable.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(repo_root / "scripts" / "export_phash_audit_reports.py"), run_name="__main__")
