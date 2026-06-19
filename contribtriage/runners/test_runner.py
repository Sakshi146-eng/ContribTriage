"""
contribtriage/runners/test_runner.py

Stage 4 — Test Verification Run.

Runs the project's test suite using the framework detected in Stage 2,
captures full output, and returns a structured TestResult.

Supported frameworks
────────────────────
  pytest        python -m pytest --tb=short -q
  unittest      python -m unittest discover -v
  cargo test    cargo test
  go test       go test ./...
  npm test      npm test
  pnpm test     pnpm test
  yarn test     yarn test

Selective rerun
───────────────
  When failed_test_ids is provided, only those tests are re-executed:
    pytest  → appends -k "test_a or test_b"
  This is a core feature of the healing loop: after a fix is applied,
  only the previously-failing tests are re-run for speed.

Design decisions
────────────────
  - Never raises: subprocess failures are caught and surfaced in TestResult.
  - Uses a configurable timeout (default 120 s) to prevent hangs.
  - Parser functions are pure (no I/O) so they are fully unit-testable.
  - dep_errors: captures import/module-not-found lines to trigger healing.
  - code_bugs: test IDs where the failure looks like a pure logic error.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console

from contribtriage.models import TestResult

console = Console()

# ===========================================================================
# Framework → Command mapping
# ===========================================================================

_COMMANDS: dict = {
    "pytest":       ["python", "-m", "pytest", "--tb=short", "-q"],
    "unittest":     ["python", "-m", "unittest", "discover", "-v"],
    "cargo test":   ["cargo", "test"],
    "go test":      ["go", "test", "./..."],
    "npm test":     ["npm", "test", "--", "--passWithNoTests"],
    "pnpm test":    ["pnpm", "test"],
    "yarn test":    ["yarn", "test"],
}

# Ecosystems each command belongs to
_ECOSYSTEM: dict = {
    "pytest":       "python",
    "unittest":     "python",
    "cargo test":   "rust",
    "go test":      "go",
    "npm test":     "node",
    "pnpm test":    "node",
    "yarn test":    "node",
}

# Default run timeout in seconds
DEFAULT_TIMEOUT: int = 120

# ===========================================================================
# Public API
# ===========================================================================

def run_tests(
    repo_root: str,
    test_framework: str,
    timeout: int = DEFAULT_TIMEOUT,
    failed_test_ids: Optional[List[str]] = None,
) -> TestResult:
    """
    Run the project's test suite and return a structured TestResult.

    Args:
        repo_root:       Absolute path to the repository root (cwd for the run).
        test_framework:  Framework key from _COMMANDS, e.g. 'pytest', 'go test'.
        timeout:         Maximum seconds to wait before killing the subprocess.
        failed_test_ids: If provided, only run these specific tests (selective
                         rerun used by the healing loop after a fix is applied).
                         Supported for ALL frameworks:
                           pytest      → -k "name_a or name_b"
                           unittest    → -k "name_a or name_b"  (via pytest runner)
                           cargo test  → positional filter (cargo test name_frag)
                           go test     → -run "TestA|TestB" regex
                           npm/pnpm/yarn → -- -t "name_a|name_b" (Jest pattern)

    Returns:
        Populated TestResult. Never raises — all errors are captured inside it.
    """
    cmd = _build_command(test_framework, failed_test_ids)
    ecosystem = _ECOSYSTEM.get(test_framework, "python")
    result = TestResult(
        source="existing",
        ecosystem=ecosystem,
        command_used=cmd,
    )

    console.print(
        f"[cyan]  → Running: {' '.join(cmd)} (timeout={timeout}s)[/cyan]"
    )

    t_start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t_start
        raw = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        result.raw_output = raw
        _parse_output(test_framework, raw, proc.returncode, result)

    except subprocess.TimeoutExpired:
        result.raw_output = f"[ContribTriage] Test run timed out after {timeout}s"
        result.errors = 1
        console.print(f"[yellow]  ⚠ Test run timed out after {timeout}s[/yellow]")

    except (OSError, FileNotFoundError) as exc:
        result.raw_output = f"[ContribTriage] Could not launch test runner: {exc}"
        result.errors = 1
        console.print(f"[red]  ✗ Could not launch '{cmd[0]}': {exc}[/red]")

    total = result.passed + result.failed + result.errors + result.skipped
    console.print(
        f"[green]  ✓ Tests: {result.passed} passed / "
        f"{result.failed} failed / {result.errors} errors "
        f"({total} total)[/green]"
    )
    return result


# ===========================================================================
# Command Builder
# ===========================================================================

def _build_command(
    framework: str,
    failed_test_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Return the shell command list for the given framework, with selective
    rerun support for ALL supported test frameworks.

    How each framework filters tests
    ────────────────────────────────
      pytest / unittest
        Appends -k "name_a or name_b" using the bare function name extracted
        from "tests/test_foo.py::TestClass::test_method" → "test_method".

      cargo test
        Appends the filter as a positional argument: `cargo test name_fragment`.
        Cargo runs tests whose full path *contains* the fragment as a substring.
        Uses the first failing test name only (cargo supports one filter token).

      go test
        Appends -run "TestA|TestB" as a regex. Extracts the test function name
        from the Go FAIL line (e.g. "--- FAIL: TestFoo" → "TestFoo").

      npm / pnpm / yarn  (Jest)
        Appends -- -t "name_a|name_b" so Jest runs matching describe/test blocks.
        The bare test name is used (last token after any :: separators).

    Falls back to the full suite command if framework is unknown.
    """
    cmd = list(_COMMANDS.get(framework, _COMMANDS["pytest"]))

    if not failed_test_ids:
        return cmd

    # ── pytest / unittest ─────────────────────────────────────────────────
    if framework in ("pytest", "unittest"):
        names = [tid.split("::")[-1] for tid in failed_test_ids if "::" in tid]
        names = names or list(failed_test_ids)
        cmd.extend(["-k", " or ".join(names)])

    # ── cargo test ────────────────────────────────────────────────────────
    elif framework == "cargo test":
        # Cargo takes a single name-substring filter as a positional argument.
        # Use the first failing test name (strip module path prefix if present).
        raw = failed_test_ids[0]
        # "tests::module::test_name" → "test_name"
        fragment = raw.split("::")[-1] if "::" in raw else raw
        cmd.append(fragment)

    # ── go test ───────────────────────────────────────────────────────────
    elif framework == "go test":
        # Extract bare TestFoo names; join as regex alternation.
        names = [tid.split("::")[-1] if "::" in tid else tid
                 for tid in failed_test_ids]
        run_pattern = "|".join(names)
        cmd.extend(["-run", run_pattern])

    # ── npm / pnpm / yarn (Jest) ──────────────────────────────────────────
    elif framework in ("npm test", "pnpm test", "yarn test"):
        names = [tid.split("::")[-1] if "::" in tid else tid
                 for tid in failed_test_ids]
        pattern = "|".join(names)
        # "--" separates npm args from the test runner args
        cmd.extend(["--", "-t", pattern])

    return cmd


# ===========================================================================
# Output Router
# ===========================================================================

def _parse_output(
    framework: str,
    raw: str,
    return_code: int,
    result: TestResult,
) -> None:
    """Dispatch to the correct parser based on framework."""
    if framework in ("pytest", "unittest"):
        _parse_pytest(raw, return_code, result)
    elif framework == "cargo test":
        _parse_cargo(raw, return_code, result)
    elif framework == "go test":
        _parse_go(raw, return_code, result)
    else:
        # npm / pnpm / yarn — parse Jest-style output
        _parse_jest(raw, return_code, result)


# ===========================================================================
# Pytest / Unittest Parser
# ===========================================================================

# e.g.  "3 failed, 10 passed, 1 warning in 2.34s"
_PY_SUMMARY   = re.compile(
    r"(?:(\d+)\s+failed)?[,\s]*"
    r"(?:(\d+)\s+passed)?[,\s]*"
    r"(?:(\d+)\s+error(?:s)?)?[,\s]*"
    r"(?:(\d+)\s+skipped)?",
    re.IGNORECASE,
)
# e.g.  "FAILED tests/test_foo.py::TestFoo::test_bar - AssertionError: ..."
_PY_FAILED    = re.compile(r"^FAILED\s+([\w/.::\-]+)\s+-\s+(.+)$", re.MULTILINE)
# Import / module errors
_PY_DEP_ERR   = re.compile(
    r"(?:ModuleNotFoundError|ImportError|No module named)\s*:?\s*(.+)",
    re.IGNORECASE,
)


def _parse_pytest(raw: str, return_code: int, result: TestResult) -> None:
    """Parse pytest / unittest output into a TestResult."""

    # ── Summary line (last match wins — final summary) ────────────────────
    summary_matches = list(_PY_SUMMARY.finditer(raw))
    for m in reversed(summary_matches):
        failed  = int(m.group(1) or 0)
        passed  = int(m.group(2) or 0)
        errors  = int(m.group(3) or 0)
        skipped = int(m.group(4) or 0)
        if passed + failed + errors + skipped > 0:
            result.passed  = passed
            result.failed  = failed
            result.errors  = errors
            result.skipped = skipped
            break

    # ── Failed test IDs ───────────────────────────────────────────────────
    for m in _PY_FAILED.finditer(raw):
        test_id = m.group(1).strip()
        short_msg = m.group(2).strip()
        result.failed_items.append((test_id, short_msg))

    # ── Dependency errors → healing trigger ───────────────────────────────
    for m in _PY_DEP_ERR.finditer(raw):
        dep_line = m.group(1).strip().strip("'\"")
        if dep_line and dep_line not in result.dep_errors:
            result.dep_errors.append(dep_line)

    # ── Code bugs: failed items that are NOT dependency errors ────────────
    dep_keywords = {"ModuleNotFoundError", "ImportError", "No module named"}
    for test_id, short_msg in result.failed_items:
        if not any(kw.lower() in short_msg.lower() for kw in dep_keywords):
            result.code_bugs.append(test_id)


# ===========================================================================
# Cargo Test Parser
# ===========================================================================

# e.g.  "test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured"
_CARGO_SUMMARY = re.compile(
    r"test result:\s+\w+\.\s+"
    r"(\d+)\s+passed;\s+(\d+)\s+failed",
    re.IGNORECASE,
)
# e.g.  "test tests::my_test ... FAILED"
_CARGO_FAILED  = re.compile(r"^test\s+([\w:]+)\s+\.\.\.\s+FAILED", re.MULTILINE)


def _parse_cargo(raw: str, return_code: int, result: TestResult) -> None:
    """Parse `cargo test` output."""
    m = _CARGO_SUMMARY.search(raw)
    if m:
        result.passed = int(m.group(1))
        result.failed = int(m.group(2))
    elif return_code == 0:
        # `cargo test` with 0 exit and no summary = compilation success, no tests
        result.passed = 0

    for m in _CARGO_FAILED.finditer(raw):
        test_id = m.group(1)
        result.failed_items.append((test_id, "FAILED"))
        result.code_bugs.append(test_id)

    # Cargo dep errors (error[E0432], etc.) are compile-time — surface them
    for line in raw.splitlines():
        if "error[" in line.lower() or "could not compile" in line.lower():
            result.dep_errors.append(line.strip())
            break


# ===========================================================================
# Go Test Parser
# ===========================================================================

# e.g.  "--- FAIL: TestFoo (0.00s)"
_GO_FAILED   = re.compile(r"^--- FAIL:\s+([\w/]+)", re.MULTILINE)
# e.g.  "ok      example.com/app    0.123s"
_GO_OK       = re.compile(r"^ok\s+", re.MULTILINE)
# e.g.  "FAIL    example.com/app    0.123s"
_GO_FAIL_PKG = re.compile(r"^FAIL\s+", re.MULTILINE)
# e.g.  "--- PASS: TestBar (0.00s)"
_GO_PASSED   = re.compile(r"^--- PASS:", re.MULTILINE)


def _parse_go(raw: str, return_code: int, result: TestResult) -> None:
    """Parse `go test ./...` output."""
    result.passed = len(_GO_PASSED.findall(raw))
    for m in _GO_FAILED.finditer(raw):
        result.failed += 1
        test_id = m.group(1)
        result.failed_items.append((test_id, "FAIL"))
        result.code_bugs.append(test_id)

    # Build/import errors
    if "cannot find package" in raw or "build failed" in raw.lower():
        result.dep_errors.append("go build error — check go.mod and imports")


# ===========================================================================
# Jest / npm Parser
# ===========================================================================

# e.g.  "Tests:       2 failed, 5 passed, 7 total"
_JEST_SUMMARY = re.compile(
    r"Tests:\s+(?:(\d+)\s+failed,\s*)?(?:(\d+)\s+passed,\s*)?(\d+)\s+total",
    re.IGNORECASE,
)
# e.g.  "  ✕ should return expected value (10 ms)"
_JEST_FAILED  = re.compile(r"^\s+[✕✗×]\s+(.+)", re.MULTILINE)


def _parse_jest(raw: str, return_code: int, result: TestResult) -> None:
    """Parse Jest / npm test output."""
    m = _JEST_SUMMARY.search(raw)
    if m:
        result.failed = int(m.group(1) or 0)
        result.passed = int(m.group(2) or 0)

    for m in _JEST_FAILED.finditer(raw):
        test_id = m.group(1).strip()
        result.failed_items.append((test_id, "FAILED"))
        result.code_bugs.append(test_id)

    # Module not found in Node
    for line in raw.splitlines():
        if "cannot find module" in line.lower() or "module not found" in line.lower():
            result.dep_errors.append(line.strip())
