"""Compatibility entry point for the renamed standalone descriptor producer.

The former methods41 production module is now codes/descriptor.py. Reuse its
authoritative tests, including independent RDM checks and exact T-axis checks.
"""
import unittest

from test_descriptor import DescriptorTests as StandaloneProductionTests


if __name__ == '__main__':
    unittest.main()
