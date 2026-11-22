import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

# Helpers that live beside the tests (magnetic_checks, imported by test_magnets.py and test_magnetic_circuits.py) are
# importable without PYTHONPATH=tests (book notes/FEEDBACK.md, 2026-11-19, 2f000a7c; the sweep cda95849).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
