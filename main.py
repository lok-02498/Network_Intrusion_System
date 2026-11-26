"""Convenience entrypoint for the Network Intrusion Detection System CLI."""

import sys
from pathlib import Path

# Allow running `python main.py` directly without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nids.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
