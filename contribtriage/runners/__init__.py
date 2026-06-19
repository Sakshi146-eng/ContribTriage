"""
contribtriage/runners/__init__.py

Stage 4 — Test Coverage Evaluation & Verification Run.

Exposes public entry points:
  - run_tests                 : Execute the test suite → TestResult
  - generate_module_test_files: Generate Groq-powered per-module test files
"""

from contribtriage.runners.test_runner import run_tests
from contribtriage.runners.test_generator import generate_module_test_files

__all__ = ["run_tests", "generate_module_test_files"]
