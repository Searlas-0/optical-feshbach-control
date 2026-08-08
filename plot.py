#!/usr/bin/env python3
"""Create the standard three figures from queried, already-saved results."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ofc.plot_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
