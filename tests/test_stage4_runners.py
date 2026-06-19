"""
tests/test_stage4_runners.py

Stage 4 test suite: Test Runner + Groq-powered Test Stub Generator.

Strategy:
  - All subprocess calls are mocked — tests are hermetic.
  - Parser functions (_parse_pytest, _parse_cargo, _parse_go, _parse_jest) are
    called directly with synthetic output strings, making them fast and reliable.
  - generate_module_test_files tests use tmp_path (pytest fixture) for real
    filesystem writes and a MagicMock groq_client for hermetic AI responses.

Coverage:
  _build_command              : command list per framework, fallback
  _parse_pytest               : passed/failed/error/skipped, FAILED lines,
                                dep errors, code_bugs classification
  _parse_cargo                : summary line, FAILED lines, compile errors
  _parse_go                   : PASS/FAIL lines, build errors
  _parse_jest                 : Tests summary line, failed items
  run_tests()                 : subprocess mock (success, failure, timeout,
                                OSError)
  generate_module_test_files  : guard cases (None KG, no uncovered, private),
                                happy path (Groq mock), fallback on bad syntax,
                                idempotency, stubs/ subdir placement,
                                one-file-per-module grouping
  _make_fallback              : per-language template correctness (Python, JS,
                                Go, Rust)
  _strip_fences               : fence removal (python, plain, none)
  _validate_syntax            : valid/invalid Python, unknown language pass-through
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from contribtriage.models import TestResult
from contribtriage.runners.test_generator import (
    generate_module_test_files,
    _make_fallback,
    _strip_fences,
    _validate_syntax,
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
# 10. generate_module_test_files() — Groq-powered per-module stub generation
# ===========================================================================

class TestGenerateModuleTestFiles:
    """Tests for the new AI-powered per-module test file generator."""

    def _make_mock_groq(self, response_text: str):
        """Build a minimal mock Groq client that returns *response_text*."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=response_text))
        ]
        return mock_client

    def _make_kg(self, uncovered_funcs, module_nodes=None):
        """Build a minimal KnowledgeGraph-like mock for testing."""
        from contribtriage.models import KnowledgeGraph, ModuleNode
        nodes = {}
        if module_nodes:
            nodes = module_nodes
        elif uncovered_funcs:
            # Auto-create a module node for each distinct module in uncovered
            from collections import defaultdict
            grouped = defaultdict(list)
            for q in uncovered_funcs:
                mod = q.rsplit(".", 1)[0] if "." in q else q
                fn  = q.rsplit(".", 1)[-1] if "." in q else q
                grouped[mod].append(fn)
            for mod, fns in grouped.items():
                nodes[mod] = ModuleNode(
                    path=f"src/{mod}.py",
                    module_name=mod,
                    language="Python",
                    functions=fns,
                    classes=[],
                    imports=["os"],
                    todos=[],
                )
        return KnowledgeGraph(
            nodes=nodes,
            edges=[],
            uncovered_funcs=uncovered_funcs,
        )

    # ── Guard cases ────────────────────────────────────────────────────────

    def test_returns_empty_list_when_kg_is_none(self, tmp_path):
        result = generate_module_test_files(
            knowledge_graph=None,
            project_meta=None,
            groq_client=None,
            repo_root=str(tmp_path),
        )
        assert result == []

    def test_returns_empty_list_when_no_uncovered(self, tmp_path):
        kg = self._make_kg([])
        result = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=None,
            repo_root=str(tmp_path),
        )
        assert result == []

    def test_skips_private_functions(self, tmp_path):
        """Functions starting with _ should not appear in uncovered stubs."""
        from contribtriage.models import KnowledgeGraph, ModuleNode
        mod_node = ModuleNode(
            path="src/utils.py",
            module_name="src.utils",
            language="Python",
            functions=["_private"],
            classes=[],
            imports=[],
            todos=[],
        )
        kg = KnowledgeGraph(
            nodes={"src.utils": mod_node},
            edges=[],
            uncovered_funcs=["src.utils._private"],  # private — should be skipped
        )
        result = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=None,
            repo_root=str(tmp_path),
        )
        assert result == []

    # ── Happy path: Groq returns valid content ─────────────────────────────

    def test_generates_file_for_uncovered_function(self, tmp_path):
        """With a mock Groq returning valid Python, one stub file is written."""
        groq_content = (
            "import pytest\n"
            "import os\n\n"
            "def test_my_func():\n"
            "    raise NotImplementedError('stub')\n"
        )
        kg     = self._make_kg(["src.utils.my_func"])
        client = self._make_mock_groq(groq_content)
        paths  = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=client,
            repo_root=str(tmp_path),
        )
        assert len(paths) == 1
        assert Path(paths[0]).exists()

    def test_generated_file_is_in_stubs_subdir(self, tmp_path):
        groq_content = "import pytest\n\ndef test_fn():\n    raise NotImplementedError()\n"
        kg     = self._make_kg(["mod.fn"])
        client = self._make_mock_groq(groq_content)
        paths  = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=client,
            repo_root=str(tmp_path),
        )
        assert "stubs" in paths[0]

    def test_generated_file_contains_function_stub(self, tmp_path):
        """Groq output is written as-is when syntax is valid."""
        groq_content = "import pytest\n\ndef test_parse_url():\n    raise NotImplementedError()\n"
        kg     = self._make_kg(["src.utils.parse_url"])
        client = self._make_mock_groq(groq_content)
        paths  = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=client,
            repo_root=str(tmp_path),
        )
        content = Path(paths[0]).read_text()
        assert "test_parse_url" in content

    def test_one_file_per_module(self, tmp_path):
        """Two functions in the same module → one file."""
        groq_content = (
            "import pytest\n\n"
            "def test_func_a():\n    raise NotImplementedError()\n\n"
            "def test_func_b():\n    raise NotImplementedError()\n"
        )
        kg     = self._make_kg(["utils.func_a", "utils.func_b"])
        client = self._make_mock_groq(groq_content)
        paths  = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=client,
            repo_root=str(tmp_path),
        )
        assert len(paths) == 1  # grouped into one file per module

    def test_two_modules_two_files(self, tmp_path):
        """Functions from different modules → separate files."""
        from contribtriage.models import KnowledgeGraph, ModuleNode
        groq_content = "import pytest\n\ndef test_fn():\n    raise NotImplementedError()\n"
        nodes = {
            "mod_a": ModuleNode("src/mod_a.py", "mod_a", "Python", [], ["fn_x"], [], []),
            "mod_b": ModuleNode("src/mod_b.py", "mod_b", "Python", [], ["fn_y"], [], []),
        }
        kg = KnowledgeGraph(
            nodes=nodes,
            edges=[],
            uncovered_funcs=["mod_a.fn_x", "mod_b.fn_y"],
        )
        client = self._make_mock_groq(groq_content)
        paths  = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=client,
            repo_root=str(tmp_path),
        )
        assert len(paths) == 2

    # ── Fallback: Groq unavailable or returns bad syntax ──────────────────

    def test_fallback_written_when_groq_client_none(self, tmp_path):
        """No groq_client → fallback template written to disk."""
        kg    = self._make_kg(["mod.my_func"])
        paths = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=None,    # no client
            repo_root=str(tmp_path),
        )
        assert len(paths) == 1
        content = Path(paths[0]).read_text()
        assert "NotImplementedError" in content or "ContribTriage" in content

    def test_fallback_written_on_groq_syntax_error(self, tmp_path):
        """Groq returns invalid Python → fallback with diagnostic comment."""
        bad_python = "def invalid syntax !!!\n    oops"
        kg     = self._make_kg(["mod.fn"])
        client = self._make_mock_groq(bad_python)
        paths  = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=client,
            repo_root=str(tmp_path),
        )
        assert len(paths) == 1
        content = Path(paths[0]).read_text()
        # Fallback content should still be valid Python
        import ast
        ast.parse(content)   # should not raise

    # ── Idempotency ───────────────────────────────────────────────────────

    def test_idempotent_does_not_overwrite_existing(self, tmp_path):
        groq_content = "import pytest\n\ndef test_fn():\n    raise NotImplementedError()\n"
        kg     = self._make_kg(["utils.my_func"])
        client = self._make_mock_groq(groq_content)
        paths1 = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=client,
            repo_root=str(tmp_path),
        )
        # Overwrite with custom sentinel
        Path(paths1[0]).write_text("# sentinel", encoding="utf-8")
        # Run again — must NOT overwrite
        paths2 = generate_module_test_files(
            knowledge_graph=kg,
            project_meta=None,
            groq_client=client,
            repo_root=str(tmp_path),
        )
        assert Path(paths2[0]).read_text() == "# sentinel"


# ===========================================================================
# 11. _make_fallback() — language-specific fallback templates
# ===========================================================================

class TestMakeFallback:
    """Tests for the per-language fallback template generator."""

    def _node(self, language="Python"):
        from contribtriage.models import ModuleNode
        return ModuleNode(
            path=f"src/mod.{'py' if language == 'Python' else 'js'}",
            module_name="mod",
            language=language,
            functions=["my_func"],
            classes=[],
            imports=[],
            todos=[],
        )

    def test_python_fallback_is_valid_python(self):
        import ast
        content = _make_fallback(self._node("Python"), ["my_func"])
        ast.parse(content)   # must not raise

    def test_python_fallback_has_not_implemented(self):
        content = _make_fallback(self._node("Python"), ["my_func"])
        assert "NotImplementedError" in content

    def test_js_fallback_has_jest_describe(self):
        content = _make_fallback(self._node("JavaScript"), ["fn"])
        assert "describe" in content
        assert "throw new Error" in content

    def test_go_fallback_has_testing_skip(self):
        content = _make_fallback(self._node("Go"), ["MyFunc"])
        assert "t.Skip" in content
        assert "testing" in content

    def test_rust_fallback_has_cfg_test(self):
        content = _make_fallback(self._node("Rust"), ["my_fn"])
        assert "#[cfg(test)]" in content
        assert "panic!" in content


# ===========================================================================
# 12. _strip_fences() and _validate_syntax()
# ===========================================================================

class TestUtilHelpers:

    def test_strip_fences_removes_python_fence(self):
        code = "```python\nimport os\n```"
        assert _strip_fences(code) == "import os"

    def test_strip_fences_removes_plain_fence(self):
        code = "```\nimport os\n```"
        assert _strip_fences(code) == "import os"

    def test_strip_fences_no_fences_unchanged(self):
        code = "import os\ndef foo(): pass"
        assert _strip_fences(code) == code

    def test_validate_syntax_valid_python(self):
        ok, msg = _validate_syntax("import os\ndef foo(): pass\n", "Python")
        assert ok is True
        assert msg == ""

    def test_validate_syntax_invalid_python(self):
        ok, msg = _validate_syntax("def invalid syntax !!!", "Python")
        assert ok is False
        assert msg != ""

    def test_validate_syntax_unknown_language_passes(self):
        """Unknown language should optimistically return True."""
        ok, _ = _validate_syntax("any content at all", "COBOL")
        assert ok is True
