"""
tests/test_stage4_runners.py

Stage 4 test suite: Test Runner + Test Stub Generator.

Strategy:
  - All subprocess calls are mocked — tests are hermetic.
  - Parser functions (_parse_pytest, _parse_cargo, _parse_go, _parse_jest) are
    called directly with synthetic output strings, making them fast and reliable.
  - test_generator tests use tmp_path (pytest fixture) for real filesystem writes.

Coverage:
  _build_command         : command list per framework, fallback
  _parse_pytest          : passed/failed/error/skipped, FAILED lines, dep errors,
                           code_bugs classification
  _parse_cargo           : summary line, FAILED lines, compile errors
  _parse_go              : PASS/FAIL lines, build errors
  _parse_jest            : Tests summary line, failed items
  run_tests()            : subprocess mock (success, failure, timeout, OSError)
  generate_test_stubs    : grouping, file content, idempotency, private-skip,
                           empty-input guard
  _make_stub             : stub function format
  _build_file            : file header content
  _module_name/_func_name: parsing helpers
  generate_dep_stubs     : Python/Node/empty input, file naming, idempotency,
                           import alias, content correctness
  _make_python_dep_stub  : import statement, pytest.fail format
  _make_node_dep_stub    : subprocess check format
  _dep_to_import_name    : alias table, hyphen normalisation
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from contribtriage.models import TestResult
from contribtriage.runners.test_generator import (
    _build_file,
    _func_name,
    _make_stub,
    _module_name,
    _make_python_dep_stub,
    _make_node_dep_stub,
    _dep_to_import_name,
    generate_test_stubs,
    generate_dep_stubs,
)
from contribtriage.runners.test_runner import (
    _build_command,
    _parse_cargo,
    _parse_go,
    _parse_jest,
    _parse_pytest,
    run_tests,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _fresh() -> TestResult:
    return TestResult()


def _mock_proc(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = returncode
    return m


# ===========================================================================
# 1. _build_command()
# ===========================================================================

class TestBuildCommand:

    def test_pytest_command(self):
        cmd = _build_command("pytest")
        assert cmd[0] == "python"
        assert "pytest" in cmd

    def test_cargo_command(self):
        cmd = _build_command("cargo test")
        assert cmd == ["cargo", "test"]

    def test_go_command(self):
        cmd = _build_command("go test")
        assert "./..." in cmd

    def test_npm_command(self):
        cmd = _build_command("npm test")
        assert "npm" in cmd

    def test_unknown_falls_back_to_pytest(self):
        cmd = _build_command("totally_unknown_framework")
        assert "pytest" in cmd

    def test_returns_new_list_each_call(self):
        """Returned list must be independent (no shared mutable state)."""
        a = _build_command("pytest")
        b = _build_command("pytest")
        a.append("extra")
        assert "extra" not in b


# ===========================================================================
# 2. _parse_pytest()
# ===========================================================================

class TestParsePytest:

    # ── Summary line parsing ────────────────────────────────────────────────

    def test_parses_passed_count(self):
        r = _fresh()
        _parse_pytest("10 passed in 1.23s", 0, r)
        assert r.passed == 10

    def test_parses_failed_count(self):
        r = _fresh()
        _parse_pytest("3 failed, 10 passed in 2.34s", 1, r)
        assert r.failed == 3
        assert r.passed == 10

    def test_parses_error_count(self):
        r = _fresh()
        _parse_pytest("1 error in 0.5s", 1, r)
        assert r.errors == 1

    def test_parses_skipped_count(self):
        r = _fresh()
        _parse_pytest("5 passed, 2 skipped in 1.0s", 0, r)
        assert r.skipped == 2

    def test_parses_all_counts_together(self):
        r = _fresh()
        _parse_pytest("2 failed, 8 passed, 1 error, 3 skipped in 4.1s", 1, r)
        assert r.failed == 2
        assert r.passed == 8
        assert r.errors == 1
        assert r.skipped == 3

    def test_zero_output_leaves_counts_zero(self):
        r = _fresh()
        _parse_pytest("", 0, r)
        assert r.passed == 0
        assert r.failed == 0

    # ── FAILED line parsing ─────────────────────────────────────────────────

    def test_parses_failed_test_id(self):
        r = _fresh()
        output = "FAILED tests/test_foo.py::TestFoo::test_bar - AssertionError: assert 1 == 2"
        _parse_pytest(output, 1, r)
        assert len(r.failed_items) == 1
        assert "tests/test_foo.py::TestFoo::test_bar" in r.failed_items[0][0]

    def test_parses_multiple_failed_items(self):
        r = _fresh()
        output = (
            "FAILED tests/test_a.py::test_one - AssertionError: x\n"
            "FAILED tests/test_b.py::test_two - ValueError: y\n"
        )
        _parse_pytest(output, 1, r)
        assert len(r.failed_items) == 2

    def test_captures_error_message_per_item(self):
        r = _fresh()
        _parse_pytest("FAILED tests/t.py::test_x - ValueError: bad value", 1, r)
        assert r.failed_items[0][1] == "ValueError: bad value"

    # ── Dependency error detection ──────────────────────────────────────────

    def test_detects_module_not_found(self):
        r = _fresh()
        _parse_pytest("ModuleNotFoundError: No module named 'requests'", 1, r)
        assert any("requests" in e for e in r.dep_errors)

    def test_detects_import_error(self):
        r = _fresh()
        _parse_pytest("ImportError: cannot import name 'foo' from 'bar'", 1, r)
        assert len(r.dep_errors) >= 1

    def test_no_dep_errors_on_clean_run(self):
        r = _fresh()
        _parse_pytest("5 passed in 1.0s", 0, r)
        assert r.dep_errors == []

    # ── Code bug classification ─────────────────────────────────────────────

    def test_assertion_error_is_code_bug(self):
        r = _fresh()
        _parse_pytest("FAILED tests/t.py::test_x - AssertionError: wrong", 1, r)
        assert "tests/t.py::test_x" in r.code_bugs

    def test_import_error_is_not_code_bug(self):
        r = _fresh()
        _parse_pytest(
            "FAILED tests/t.py::test_x - ModuleNotFoundError: No module named 'foo'",
            1, r,
        )
        assert "tests/t.py::test_x" not in r.code_bugs


# ===========================================================================
# 3. _parse_cargo()
# ===========================================================================

class TestParseCargo:

    SUMMARY = "test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured"
    FAILED_LINE = "test tests::my_function ... FAILED"

    def test_parses_passed(self):
        r = _fresh()
        _parse_cargo(self.SUMMARY, 1, r)
        assert r.passed == 2

    def test_parses_failed(self):
        r = _fresh()
        _parse_cargo(self.SUMMARY, 1, r)
        assert r.failed == 1

    def test_detects_failed_test_ids(self):
        r = _fresh()
        _parse_cargo(self.FAILED_LINE + "\n" + self.SUMMARY, 1, r)
        assert ("tests::my_function", "FAILED") in r.failed_items

    def test_failed_id_added_to_code_bugs(self):
        r = _fresh()
        _parse_cargo(self.FAILED_LINE, 1, r)
        assert "tests::my_function" in r.code_bugs

    def test_compile_error_adds_dep_error(self):
        r = _fresh()
        _parse_cargo("error[E0432]: unresolved import `serde`", 101, r)
        assert len(r.dep_errors) >= 1

    def test_clean_run_no_errors(self):
        r = _fresh()
        _parse_cargo("test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured", 0, r)
        assert r.failed == 0
        assert r.dep_errors == []


# ===========================================================================
# 4. _parse_go()
# ===========================================================================

class TestParseGo:

    PASS_OUTPUT = "--- PASS: TestFoo (0.00s)\n--- PASS: TestBar (0.01s)\nok example.com/app 0.05s"
    FAIL_OUTPUT = "--- FAIL: TestBaz (0.00s)\nFAIL\texample.com/app\t0.05s"

    def test_counts_passing_tests(self):
        r = _fresh()
        _parse_go(self.PASS_OUTPUT, 0, r)
        assert r.passed == 2

    def test_counts_failing_tests(self):
        r = _fresh()
        _parse_go(self.FAIL_OUTPUT, 1, r)
        assert r.failed == 1

    def test_failed_id_in_failed_items(self):
        r = _fresh()
        _parse_go(self.FAIL_OUTPUT, 1, r)
        assert any("TestBaz" in item[0] for item in r.failed_items)

    def test_go_build_error_triggers_dep_error(self):
        r = _fresh()
        _parse_go("cannot find package: github.com/some/pkg", 1, r)
        assert len(r.dep_errors) >= 1

    def test_clean_run_no_failures(self):
        r = _fresh()
        _parse_go(self.PASS_OUTPUT, 0, r)
        assert r.failed == 0
        assert r.dep_errors == []


# ===========================================================================
# 5. _parse_jest()
# ===========================================================================

class TestParseJest:

    SUMMARY = "Tests:       2 failed, 5 passed, 7 total"

    def test_parses_failed_count(self):
        r = _fresh()
        _parse_jest(self.SUMMARY, 1, r)
        assert r.failed == 2

    def test_parses_passed_count(self):
        r = _fresh()
        _parse_jest(self.SUMMARY, 1, r)
        assert r.passed == 5

    def test_detects_failed_test_line(self):
        r = _fresh()
        output = "  ✕ should return expected value (10 ms)\n" + self.SUMMARY
        _parse_jest(output, 1, r)
        assert len(r.failed_items) >= 1

    def test_module_not_found_triggers_dep_error(self):
        r = _fresh()
        _parse_jest("Cannot find module 'some-pkg' from 'src/index.js'", 1, r)
        assert len(r.dep_errors) >= 1

    def test_clean_run_no_errors(self):
        r = _fresh()
        _parse_jest("Tests:       0 failed, 10 passed, 10 total", 0, r)
        assert r.failed == 0
        assert r.dep_errors == []


# ===========================================================================
# 6. run_tests() Integration
# ===========================================================================

class TestRunTests:

    def _mock_run(self, stdout: str, returncode: int = 0):
        return patch(
            "contribtriage.runners.test_runner.subprocess.run",
            return_value=_mock_proc(stdout=stdout, returncode=returncode),
        )

    def test_returns_test_result_type(self, tmp_path):
        with self._mock_run("5 passed in 1.0s", 0):
            result = run_tests(str(tmp_path), "pytest")
        assert isinstance(result, TestResult)

    def test_passed_count_populated(self, tmp_path):
        with self._mock_run("7 passed in 2.0s", 0):
            result = run_tests(str(tmp_path), "pytest")
        assert result.passed == 7

    def test_failed_count_populated(self, tmp_path):
        with self._mock_run("3 failed, 7 passed in 2.0s", 1):
            result = run_tests(str(tmp_path), "pytest")
        assert result.failed == 3

    def test_raw_output_captured(self, tmp_path):
        with self._mock_run("10 passed in 1.0s", 0):
            result = run_tests(str(tmp_path), "pytest")
        assert "10 passed" in result.raw_output

    def test_command_used_is_set(self, tmp_path):
        with self._mock_run("0 passed in 0.1s", 0):
            result = run_tests(str(tmp_path), "pytest")
        assert len(result.command_used) > 0

    def test_ecosystem_set_for_cargo(self, tmp_path):
        with self._mock_run("test result: ok. 0 passed; 0 failed", 0):
            result = run_tests(str(tmp_path), "cargo test")
        assert result.ecosystem == "rust"

    def test_timeout_captured_gracefully(self, tmp_path):
        with patch(
            "contribtriage.runners.test_runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired("pytest", 120),
        ):
            result = run_tests(str(tmp_path), "pytest", timeout=120)
        assert result.errors >= 1

    def test_os_error_captured_gracefully(self, tmp_path):
        with patch(
            "contribtriage.runners.test_runner.subprocess.run",
            side_effect=OSError("no such file"),
        ):
            result = run_tests(str(tmp_path), "pytest")
        assert result.errors >= 1


# ===========================================================================
# 7. _module_name() and _func_name()
# ===========================================================================

class TestNameHelpers:

    def test_module_name_simple(self):
        assert _module_name("src.utils.parse_url") == "src.utils"

    def test_module_name_single_level(self):
        assert _module_name("utils.my_func") == "utils"

    def test_func_name_simple(self):
        assert _func_name("src.utils.parse_url") == "parse_url"

    def test_func_name_single_level(self):
        assert _func_name("utils.my_func") == "my_func"


# ===========================================================================
# 8. _make_stub()
# ===========================================================================

class TestMakeStub:

    def test_stub_is_valid_function_def(self):
        stub = _make_stub("src.utils.parse_url")
        assert "def test_parse_url():" in stub

    def test_stub_contains_qualified_name(self):
        stub = _make_stub("src.utils.parse_url")
        assert "src.utils.parse_url" in stub

    def test_stub_raises_not_implemented(self):
        stub = _make_stub("src.utils.parse_url")
        assert "NotImplementedError" in stub

    def test_stub_has_docstring(self):
        stub = _make_stub("mymod.myfunc")
        assert '"""' in stub


# ===========================================================================
# 9. _build_file()
# ===========================================================================

class TestBuildFile:

    def test_header_contains_module_name(self):
        content = _build_file("src.utils", ["def test_foo(): pass\n"])
        assert "src.utils" in content

    def test_header_contains_stub_count(self):
        stubs = ["def test_a(): pass\n", "def test_b(): pass\n"]
        content = _build_file("mymod", stubs)
        assert "2" in content  # stub count mentioned

    def test_file_imports_pytest(self):
        content = _build_file("mod", ["def test_x(): pass\n"])
        assert "import pytest" in content


# ===========================================================================
# 10. generate_test_stubs() Integration
# ===========================================================================

class TestGenerateTestStubs:

    def test_returns_empty_list_when_no_uncovered(self, tmp_path):
        result = generate_test_stubs([], str(tmp_path))
        assert result == []

    def test_skips_private_functions(self, tmp_path):
        result = generate_test_stubs(["mod._private_func"], str(tmp_path))
        assert result == []

    def test_generates_file_for_uncovered_function(self, tmp_path):
        paths = generate_test_stubs(["src.utils.parse_url"], str(tmp_path))
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_generated_file_in_stubs_subdir(self, tmp_path):
        paths = generate_test_stubs(["mod.my_func"], str(tmp_path))
        assert "stubs" in paths[0]

    def test_file_contains_stub_function(self, tmp_path):
        paths = generate_test_stubs(["src.utils.parse_url"], str(tmp_path))
        content = Path(paths[0]).read_text()
        assert "def test_parse_url" in content

    def test_functions_from_same_module_in_one_file(self, tmp_path):
        paths = generate_test_stubs(
            ["utils.func_a", "utils.func_b"], str(tmp_path)
        )
        assert len(paths) == 1  # grouped into one file
        content = Path(paths[0]).read_text()
        assert "def test_func_a" in content
        assert "def test_func_b" in content

    def test_functions_from_different_modules_in_separate_files(self, tmp_path):
        paths = generate_test_stubs(
            ["mod_a.func_x", "mod_b.func_y"], str(tmp_path)
        )
        assert len(paths) == 2

    def test_idempotent_does_not_overwrite_existing(self, tmp_path):
        uncovered = ["utils.my_func"]
        paths1 = generate_test_stubs(uncovered, str(tmp_path))
        # Overwrite the file with custom content
        Path(paths1[0]).write_text("# custom content", encoding="utf-8")
        # Run again — should NOT overwrite
        paths2 = generate_test_stubs(uncovered, str(tmp_path))
        assert Path(paths2[0]).read_text() == "# custom content"

    def test_creates_init_in_stubs_dir(self, tmp_path):
        generate_test_stubs(["utils.my_func"], str(tmp_path))
        init = tmp_path / "tests" / "stubs" / "__init__.py"
        assert init.exists()

    def test_generated_content_raises_not_implemented(self, tmp_path):
        paths = generate_test_stubs(["utils.my_func"], str(tmp_path))
        content = Path(paths[0]).read_text()
        assert "NotImplementedError" in content


# ===========================================================================
# 11. _dep_to_import_name()
# ===========================================================================

class TestDepNameHelpers:

    def test_plain_name_unchanged(self):
        assert _dep_to_import_name("requests") == "requests"

    def test_hyphen_replaced_by_underscore(self):
        assert _dep_to_import_name("my-cool-lib") == "my_cool_lib"

    def test_alias_pillow(self):
        assert _dep_to_import_name("pillow") == "PIL"

    def test_alias_scikit_learn(self):
        assert _dep_to_import_name("scikit-learn") == "sklearn"

    def test_alias_pyyaml(self):
        assert _dep_to_import_name("pyyaml") == "yaml"

    def test_alias_beautifulsoup4(self):
        assert _dep_to_import_name("beautifulsoup4") == "bs4"

    def test_alias_python_dotenv(self):
        assert _dep_to_import_name("python-dotenv") == "dotenv"

    def test_alias_lookup_is_case_insensitive(self):
        # Alias table keys are lowercase; input may be mixed case
        assert _dep_to_import_name("Pillow") == "PIL"
        assert _dep_to_import_name("PYYAML") == "yaml"


# ===========================================================================
# 12. _make_python_dep_stub()
# ===========================================================================

class TestMakePythonDepStub:

    def test_stub_has_function_def(self):
        stub = _make_python_dep_stub("requests")
        assert "def test_import_requests():" in stub

    def test_stub_has_import_statement(self):
        stub = _make_python_dep_stub("requests")
        assert "import requests" in stub

    def test_stub_has_pytest_fail(self):
        stub = _make_python_dep_stub("requests")
        assert "pytest.fail" in stub

    def test_stub_uses_alias_for_pillow(self):
        stub = _make_python_dep_stub("pillow")
        assert "import PIL" in stub

    def test_stub_uses_safe_name_for_hyphenated_dep(self):
        """Hyphens in dep name must become underscores in the function name."""
        stub = _make_python_dep_stub("python-dotenv")
        assert "def test_import_python_dotenv" in stub

    def test_stub_has_docstring(self):
        stub = _make_python_dep_stub("requests")
        assert '"""' in stub


# ===========================================================================
# 13. _make_node_dep_stub()
# ===========================================================================

class TestMakeNodeDepStub:

    def test_stub_has_function_def(self):
        stub = _make_node_dep_stub("axios")
        assert "def test_import_axios():" in stub

    def test_stub_runs_node_subprocess(self):
        stub = _make_node_dep_stub("axios")
        assert '"node"' in stub
        assert "require" in stub

    def test_stub_has_pytest_fail(self):
        stub = _make_node_dep_stub("axios")
        assert "pytest.fail" in stub

    def test_scoped_package_safe_name(self):
        """@scope/pkg names must produce a valid Python identifier for the def line.
        The @ stays inside the require() call — that's intentional npm syntax."""
        stub = _make_node_dep_stub("@scope/my-pkg")
        # Function name must be a valid identifier (no @ or / characters)
        first_line = stub.splitlines()[0]
        assert "def test_import_" in first_line
        assert "@" not in first_line
        assert "/" not in first_line
        # But the require() call MUST keep the original package name
        assert "require('@scope/my-pkg')" in stub


# ===========================================================================
# 14. generate_dep_stubs() Integration
# ===========================================================================

class TestGenerateDepStubs:

    # Shared fixture helpers
    _PY_DEPS = ["requests", "flask"]
    _PY_ECO  = {"requests": "python", "flask": "python"}
    _NODE_DEPS = ["axios", "express"]
    _NODE_ECO  = {"axios": "node", "express": "node"}

    def test_returns_empty_list_when_no_deps(self, tmp_path):
        result = generate_dep_stubs([], {}, str(tmp_path))
        assert result == []

    def test_generates_python_dep_file(self, tmp_path):
        paths = generate_dep_stubs(self._PY_DEPS, self._PY_ECO, str(tmp_path))
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_python_dep_file_has_correct_name(self, tmp_path):
        paths = generate_dep_stubs(self._PY_DEPS, self._PY_ECO, str(tmp_path))
        assert "test_deps_python_stubs.py" in paths[0]

    def test_generates_node_dep_file(self, tmp_path):
        paths = generate_dep_stubs(self._NODE_DEPS, self._NODE_ECO, str(tmp_path))
        assert len(paths) == 1
        assert "test_deps_node_stubs.py" in paths[0]

    def test_generates_both_files_for_mixed_repo(self, tmp_path):
        deps = self._PY_DEPS + self._NODE_DEPS
        eco  = {**self._PY_ECO, **self._NODE_ECO}
        paths = generate_dep_stubs(deps, eco, str(tmp_path))
        assert len(paths) == 2

    def test_rust_and_go_deps_are_skipped(self, tmp_path):
        deps = ["serde", "tokio"]
        eco  = {"serde": "rust", "tokio": "rust"}
        paths = generate_dep_stubs(deps, eco, str(tmp_path))
        assert paths == []

    def test_python_file_contains_import_test_per_dep(self, tmp_path):
        paths = generate_dep_stubs(self._PY_DEPS, self._PY_ECO, str(tmp_path))
        content = Path(paths[0]).read_text()
        assert "def test_import_requests" in content
        assert "def test_import_flask" in content

    def test_node_file_contains_require_test_per_dep(self, tmp_path):
        paths = generate_dep_stubs(self._NODE_DEPS, self._NODE_ECO, str(tmp_path))
        content = Path(paths[0]).read_text()
        assert "require" in content
        assert "def test_import_axios" in content

    def test_python_file_imports_pytest(self, tmp_path):
        paths = generate_dep_stubs(self._PY_DEPS, self._PY_ECO, str(tmp_path))
        content = Path(paths[0]).read_text()
        assert "import pytest" in content

    def test_file_in_stubs_subdir(self, tmp_path):
        paths = generate_dep_stubs(self._PY_DEPS, self._PY_ECO, str(tmp_path))
        assert "stubs" in paths[0]

    def test_creates_init_file_in_stubs_dir(self, tmp_path):
        generate_dep_stubs(self._PY_DEPS, self._PY_ECO, str(tmp_path))
        init = tmp_path / "tests" / "stubs" / "__init__.py"
        assert init.exists()

    def test_idempotent_python_file(self, tmp_path):
        """Calling twice must NOT overwrite the first file."""
        paths1 = generate_dep_stubs(self._PY_DEPS, self._PY_ECO, str(tmp_path))
        Path(paths1[0]).write_text("# sentinel", encoding="utf-8")
        paths2 = generate_dep_stubs(self._PY_DEPS, self._PY_ECO, str(tmp_path))
        assert Path(paths2[0]).read_text() == "# sentinel"

    def test_uses_import_alias_in_generated_test(self, tmp_path):
        """Pillow's generated import should be 'PIL', not 'pillow'."""
        deps = ["pillow"]
        eco  = {"pillow": "python"}
        paths = generate_dep_stubs(deps, eco, str(tmp_path))
        content = Path(paths[0]).read_text()
        assert "import PIL" in content
        assert "import pillow" not in content

    def test_dep_with_unknown_ecosystem_skipped(self, tmp_path):
        """A dep with ecosystem='unknown' should produce no files."""
        paths = generate_dep_stubs(
            ["some-tool"], {"some-tool": "unknown"}, str(tmp_path)
        )
        assert paths == []
