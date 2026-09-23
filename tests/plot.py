#!/usr/bin/env python3
"""Compatibility launcher for the current Part 3 / 30-point RHF overlay.

Uses codes/plots/plot_rhf_bond_scan_30.py and its zero-criterion metadata.
Only saved RHF data are plotted; no electronic calculations are launched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))
from plots.plot_rhf_bond_scan_30 import main

if __name__ == "__main__":
    raise SystemExit(main())
