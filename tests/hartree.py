#!/usr/bin/env python3
"""Regenerate Hartree outputs from the saved scan using an easy-to-copy filename."""

import sys
from pathlib import Path

import rhf_bound_level_selection as workflow


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    saved_scan = Path(__file__).resolve().parent / "rhf_bound_level_selection.json"
    return workflow.main(["--reuse-scan", str(saved_scan), *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
