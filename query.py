#!/usr/bin/env python3
"""Independent read-only result search; it cannot launch calculations or plots."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ofc.query_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
