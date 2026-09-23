#!/usr/bin/env python3
"""Regenerate current zero-energy Part 3 outputs from saved Parts 1/2."""

import sys

import rhf_bound_level_selection as workflow


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    return workflow.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
