"""
contribtriage/runners/__init__.py

Stage 4 — Test Coverage Evaluation & Verification Run.

Exposes three public entry points:
  - run_tests           : Execute the test suite → TestResult
  - generate_test_stubs : Emit stub files for uncovered public functions
  - generate_dep_stubs  : Emit importability-check tests for declared deps
"""

from contribtriage.runners.test_runner import run_tests
from contribtriage.runners.test_generator import generate_test_stubs, generate_dep_stubs

__all__ = ["run_tests", "generate_test_stubs", "generate_dep_stubs"]

