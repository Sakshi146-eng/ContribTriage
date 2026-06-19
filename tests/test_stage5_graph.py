"""
tests/test_stage5_graph.py

Stage 5 test suite: LangGraph Nodes + Conditional Edge Routing.

Strategy:
  - Edge routing functions (edges.py) are pure — tested as plain functions.
  - Node functions (nodes.py) have all service calls mocked (test_runner,
    groq_agent, installer, gemini_agent) — hermetic and fast.
  - Groq client is always a MagicMock — no real API calls.
  - No LangGraph graph.invoke() in these tests — nodes are tested directly
    as functions returning partial state dicts.

Coverage:
  route_coverage      : KG uncovered funcs / declared deps / no KG
  route_test_result   : passed, failed, no result
  route_failure       : code_bug, app_dep, system_dep, unknown
  route_consent       : skipped_to_report, max_retries, retry ok
  run_tests_node      : result populated, terminal_log_history appended,
                        failed_test_ids updated, selective rerun flag
  analyze_failure_node: merges history, updates failure_category + fix_command
  apply_fix_node      : user declined, success, failure, retry_count increment,
                        dep_findings populated, skipped_to_report flag
  report_node         : calls generate_report, returns final_report_path
  _build_command      : selective rerun -k expression
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from contribtriage.models import (
    DependencyFinding,
    DepStatus,
    EnvReport,
    KnowledgeGraph,
    ProjectMeta,
    TestResult,
)
from contribtriage.graph.edges import (
    route_consent,
    route_coverage,
    route_failure,
    route_test_result,
)
from contribtriage.graph.nodes import (
    analyze_failure_node,
    apply_fix_node,
    report_node,
    run_tests_node,
    _infer_dep_type,
)
from contribtriage.runners.test_runner import _build_command


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def base_state() -> Dict[str, Any]:
    """Minimal valid LangGraphState for most tests."""
    env = EnvReport(
        os_name="Linux",
        os_version="5.15.0",
        python_version="3.11.4",
        python_path="/usr/bin/python3",
        docker_running=True,
        uv_available=True,
        pip_available=True,
        npm_available=True,
    )
    meta = ProjectMeta()
    meta.declared_deps = ["requests"]
    meta.dep_ecosystem = {"requests": "python"}
    meta.test_framework = ["pytest"]

    return {
        "repo_path":            "/tmp/testrepo",
        "yes_flag":             True,
        "max_retries":          3,
        "retry_count":          0,
        "terminal_log_history": [],
        "groq_analysis":        "",
        "failed_test_ids":      [],
        "skipped_to_report":    False,
        "failure_category":     "",
        "fix_command":          "",
        "env_report":           env,
        "project_meta":         meta,
        "groq_client":          MagicMock(),
        "dep_findings":         [],
        "test_result":          None,
        "final_report_path":    "",
        "knowledge_graph":      None,
    }


def _make_test_result(passed=5, failed=0, errors=0, items=None) -> TestResult:
    r = TestResult()
    r.passed = passed
    r.failed = failed
    r.errors = errors
    r.failed_items = items or []
    r.command_used = ["python", "-m", "pytest"]
    r.raw_output = f"{passed} passed, {failed} failed"
    return r


# ===========================================================================
# 1. route_coverage()
# ===========================================================================

class TestRouteCoverage:

    def test_no_kg_routes_to_run_tests(self, base_state):
        base_state["knowledge_graph"] = None
        assert route_coverage(base_state) == "run_tests"

    def test_uncovered_funcs_routes_to_generate(self, base_state):
        kg = MagicMock()
        kg.uncovered_funcs = ["mod.func_a", "mod.func_b"]
        base_state["knowledge_graph"] = kg
        assert route_coverage(base_state) == "generate_tests"

    def test_declared_deps_routes_to_generate(self, base_state):
        kg = MagicMock()
        kg.uncovered_funcs = []
        base_state["knowledge_graph"] = kg
        # project_meta has declared_deps = ["requests"]
        assert route_coverage(base_state) == "generate_tests"

    def test_empty_uncovered_and_no_deps_routes_to_run(self, base_state):
        kg = MagicMock()
        kg.uncovered_funcs = []
        base_state["knowledge_graph"] = kg
        base_state["project_meta"].declared_deps = []
        assert route_coverage(base_state) == "run_tests"


# ===========================================================================
# 2. route_test_result()
# ===========================================================================

class TestRouteTestResult:

    def test_all_passed_routes_to_report(self, base_state):
        base_state["test_result"] = _make_test_result(passed=5, failed=0, errors=0)
        assert route_test_result(base_state) == "report"

    def test_failures_route_to_analyze(self, base_state):
        base_state["test_result"] = _make_test_result(passed=3, failed=2)
        assert route_test_result(base_state) == "analyze_failure"

    def test_errors_route_to_analyze(self, base_state):
        r = _make_test_result()
        r.errors = 1
        base_state["test_result"] = r
        assert route_test_result(base_state) == "analyze_failure"

    def test_no_result_routes_to_analyze(self, base_state):
        base_state["test_result"] = None
        assert route_test_result(base_state) == "analyze_failure"


# ===========================================================================
# 3. route_failure()
# ===========================================================================

class TestRouteFailure:

    def test_code_bug_routes_to_report(self, base_state):
        base_state["failure_category"] = "code_bug"
        assert route_failure(base_state) == "report"

    def test_app_dep_routes_to_apply_fix(self, base_state):
        base_state["failure_category"] = "app_dep"
        assert route_failure(base_state) == "apply_fix"

    def test_system_dep_routes_to_apply_fix(self, base_state):
        base_state["failure_category"] = "system_dep"
        assert route_failure(base_state) == "apply_fix"

    def test_empty_category_routes_to_report(self, base_state):
        base_state["failure_category"] = ""
        assert route_failure(base_state) == "report"

    def test_unknown_category_routes_to_report(self, base_state):
        base_state["failure_category"] = "alien_dep"
        assert route_failure(base_state) == "report"


# ===========================================================================
# 4. route_consent()
# ===========================================================================

class TestRouteConsent:

    def test_skipped_to_report_routes_to_report(self, base_state):
        base_state["skipped_to_report"] = True
        assert route_consent(base_state) == "report"

    def test_max_retries_exhausted_routes_to_report(self, base_state):
        base_state["retry_count"] = 3
        base_state["max_retries"] = 3
        assert route_consent(base_state) == "report"

    def test_retry_available_routes_to_run_tests(self, base_state):
        base_state["retry_count"] = 1
        base_state["max_retries"] = 3
        base_state["skipped_to_report"] = False
        assert route_consent(base_state) == "run_tests"

    def test_zero_retries_routes_to_run_tests(self, base_state):
        base_state["retry_count"] = 0
        assert route_consent(base_state) == "run_tests"


# ===========================================================================
# 5. _build_command() — selective rerun across ALL frameworks
# ===========================================================================

class TestBuildCommandSelective:

    # ── pytest ────────────────────────────────────────────────────────────

    def test_no_failed_ids_returns_standard_command(self):
        cmd = _build_command("pytest", failed_test_ids=[])
        assert "-k" not in cmd

    def test_empty_list_returns_standard_command(self):
        cmd = _build_command("pytest", failed_test_ids=[])
        assert cmd == ["python", "-m", "pytest", "--tb=short", "-q"]

    def test_failed_ids_adds_k_flag(self):
        ids = ["tests/test_foo.py::TestClass::test_bar"]
        cmd = _build_command("pytest", failed_test_ids=ids)
        assert "-k" in cmd

    def test_k_expression_uses_function_name(self):
        ids = ["tests/test_foo.py::TestClass::test_bar"]
        cmd = _build_command("pytest", failed_test_ids=ids)
        k_idx = cmd.index("-k")
        assert "test_bar" in cmd[k_idx + 1]

    def test_multiple_ids_joined_with_or(self):
        ids = [
            "tests/test_foo.py::TestA::test_x",
            "tests/test_foo.py::TestA::test_y",
        ]
        cmd = _build_command("pytest", failed_test_ids=ids)
        k_idx = cmd.index("-k")
        expr = cmd[k_idx + 1]
        assert "test_x" in expr
        assert "or" in expr
        assert "test_y" in expr

    def test_none_failed_ids_no_k_flag(self):
        cmd = _build_command("pytest", failed_test_ids=None)
        assert "-k" not in cmd

    # ── cargo test ────────────────────────────────────────────────────────

    def test_cargo_appends_positional_filter(self):
        """cargo test uses a positional name-fragment, not -k."""
        ids = ["tests::module::test_my_fn"]
        cmd = _build_command("cargo test", failed_test_ids=ids)
        assert "test_my_fn" in cmd
        assert "-k" not in cmd

    def test_cargo_strips_module_prefix(self):
        ids = ["tests::integration::test_connect"]
        cmd = _build_command("cargo test", failed_test_ids=ids)
        # bare function name after last :: should be the positional arg
        assert "test_connect" in cmd
        assert "tests::integration::test_connect" not in " ".join(cmd)

    def test_cargo_no_ids_returns_standard_command(self):
        cmd = _build_command("cargo test", failed_test_ids=None)
        assert cmd == ["cargo", "test"]

    # ── go test ───────────────────────────────────────────────────────────

    def test_go_test_appends_run_flag(self):
        ids = ["TestFooBar"]
        cmd = _build_command("go test", failed_test_ids=ids)
        assert "-run" in cmd

    def test_go_test_run_pattern_single(self):
        ids = ["TestFooBar"]
        cmd = _build_command("go test", failed_test_ids=ids)
        run_idx = cmd.index("-run")
        assert cmd[run_idx + 1] == "TestFooBar"

    def test_go_test_run_pattern_multiple_joined_with_pipe(self):
        ids = ["TestAlpha", "TestBeta"]
        cmd = _build_command("go test", failed_test_ids=ids)
        run_idx = cmd.index("-run")
        pattern = cmd[run_idx + 1]
        assert "TestAlpha" in pattern
        assert "|" in pattern
        assert "TestBeta" in pattern

    def test_go_test_no_ids_returns_standard_command(self):
        cmd = _build_command("go test", failed_test_ids=None)
        assert cmd == ["go", "test", "./..."]

    # ── npm / pnpm / yarn (Jest) ──────────────────────────────────────────

    def test_npm_test_appends_jest_pattern(self):
        ids = ["should return 200"]
        cmd = _build_command("npm test", failed_test_ids=ids)
        assert "--" in cmd
        assert "-t" in cmd

    def test_npm_test_pattern_value(self):
        ids = ["should return 200"]
        cmd = _build_command("npm test", failed_test_ids=ids)
        t_idx = cmd.index("-t")
        assert "should return 200" in cmd[t_idx + 1]

    def test_npm_test_multiple_ids_joined_with_pipe(self):
        ids = ["test_alpha", "test_beta"]
        cmd = _build_command("npm test", failed_test_ids=ids)
        t_idx = cmd.index("-t")
        pattern = cmd[t_idx + 1]
        assert "test_alpha" in pattern
        assert "|" in pattern
        assert "test_beta" in pattern

    def test_pnpm_test_appends_jest_pattern(self):
        ids = ["should work"]
        cmd = _build_command("pnpm test", failed_test_ids=ids)
        assert "--" in cmd
        assert "-t" in cmd

    def test_yarn_test_appends_jest_pattern(self):
        ids = ["should work"]
        cmd = _build_command("yarn test", failed_test_ids=ids)
        assert "--" in cmd
        assert "-t" in cmd

    def test_npm_test_no_ids_returns_standard_command(self):
        # Base npm test command already contains "--passWithNoTests",
        # so we check that the Jest -t filter was NOT added.
        cmd = _build_command("npm test", failed_test_ids=None)
        assert "-t" not in cmd


# ===========================================================================
# 6. run_tests_node()
# ===========================================================================

class TestRunTestsNode:

    def test_returns_test_result(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        mock_result = _make_test_result(passed=7, failed=0)
        with patch("contribtriage.graph.nodes.run_tests", return_value=mock_result):
            update = run_tests_node(base_state)
        assert update["test_result"].passed == 7

    def test_appends_to_terminal_log_history(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        mock_result = _make_test_result()
        with patch("contribtriage.graph.nodes.run_tests", return_value=mock_result):
            update = run_tests_node(base_state)
        assert len(update["terminal_log_history"]) == 1
        assert "Test Run" in update["terminal_log_history"][0]

    def test_log_entry_contains_passed_count(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        mock_result = _make_test_result(passed=9)
        with patch("contribtriage.graph.nodes.run_tests", return_value=mock_result):
            update = run_tests_node(base_state)
        assert "9" in update["terminal_log_history"][0]

    def test_updates_failed_test_ids(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        r = _make_test_result(failed=1)
        r.failed_items = [("tests/test_foo.py::test_bar", "AssertionError")]
        with patch("contribtriage.graph.nodes.run_tests", return_value=r):
            update = run_tests_node(base_state)
        assert "tests/test_foo.py::test_bar" in update["failed_test_ids"]

    def test_passes_failed_ids_for_selective_rerun(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["failed_test_ids"] = ["tests/test_foo.py::test_bar"]
        mock_result = _make_test_result()
        with patch("contribtriage.graph.nodes.run_tests", return_value=mock_result) as mock_run:
            run_tests_node(base_state)
        _, kwargs = mock_run.call_args
        assert kwargs.get("failed_test_ids") == ["tests/test_foo.py::test_bar"]


# ===========================================================================
# 7. analyze_failure_node()
# ===========================================================================

class TestAnalyzeFailureNode:

    def _mock_analysis(self, category="app_dep", fix_cmd="pip install x"):
        return {
            "failure_category": category,
            "reasoning": f"test reasoning for {category}",
            "fix_command": fix_cmd,
            "fix_description": "fixes the dep",
            "affected_packages": ["x"],
        }

    def test_merges_full_history_before_calling_groq(self, base_state):
        base_state["terminal_log_history"] = ["log1", "log2", "log3"]
        with patch("contribtriage.graph.nodes.analyze_failure",
                   return_value=self._mock_analysis()) as mock_groq:
            analyze_failure_node(base_state)
        call_kwargs = mock_groq.call_args[1]
        raw = call_kwargs.get("raw_output", mock_groq.call_args[0][0] if mock_groq.call_args[0] else "")
        assert "log1" in raw
        assert "log2" in raw
        assert "log3" in raw

    def test_sets_failure_category(self, base_state):
        base_state["terminal_log_history"] = ["some failure"]
        with patch("contribtriage.graph.nodes.analyze_failure",
                   return_value=self._mock_analysis("system_dep", "docker run postgres")):
            update = analyze_failure_node(base_state)
        assert update["failure_category"] == "system_dep"

    def test_sets_fix_command(self, base_state):
        base_state["terminal_log_history"] = ["error"]
        with patch("contribtriage.graph.nodes.analyze_failure",
                   return_value=self._mock_analysis("app_dep", "pip install flask")):
            update = analyze_failure_node(base_state)
        assert update["fix_command"] == "pip install flask"

    def test_sets_groq_analysis(self, base_state):
        base_state["terminal_log_history"] = ["error"]
        analysis = self._mock_analysis()
        analysis["reasoning"] = "requests package is not installed"
        with patch("contribtriage.graph.nodes.analyze_failure", return_value=analysis):
            update = analyze_failure_node(base_state)
        assert "requests" in update["groq_analysis"]

    def test_appends_to_terminal_log_history(self, base_state):
        base_state["terminal_log_history"] = ["existing"]
        with patch("contribtriage.graph.nodes.analyze_failure",
                   return_value=self._mock_analysis()):
            update = analyze_failure_node(base_state)
        assert len(update["terminal_log_history"]) == 1
        assert "Groq Analysis" in update["terminal_log_history"][0]

    def test_empty_fix_command_stored_as_empty_string(self, base_state):
        base_state["terminal_log_history"] = ["error"]
        analysis = self._mock_analysis("code_bug", None)
        with patch("contribtriage.graph.nodes.analyze_failure", return_value=analysis):
            update = analyze_failure_node(base_state)
        assert update["fix_command"] == ""


# ===========================================================================
# 8. apply_fix_node()
# ===========================================================================

class TestApplyFixNode:

    def test_user_decline_sets_skipped_to_report(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "pip install requests"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(False, "User declined.", True)):
            update = apply_fix_node(base_state)
        assert update["skipped_to_report"] is True

    def test_user_decline_does_not_increment_retry(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["retry_count"] = 0
        base_state["fix_command"] = "pip install requests"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(False, "declined", True)):
            update = apply_fix_node(base_state)
        assert update["retry_count"] == 1  # still increments for bookkeeping

    def test_successful_fix_increments_retry_count(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["retry_count"] = 1
        base_state["fix_command"] = "pip install requests"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(True, "Successfully installed", False)):
            update = apply_fix_node(base_state)
        assert update["retry_count"] == 2

    def test_success_records_installed_status(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "pip install requests"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(True, "ok", False)):
            update = apply_fix_node(base_state)
        finding = update["dep_findings"][0]
        assert finding.status == DepStatus.INSTALLED

    def test_failure_records_failed_status(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "pip install bad-pkg"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(False, "error", False)):
            update = apply_fix_node(base_state)
        finding = update["dep_findings"][0]
        assert finding.status == DepStatus.FAILED

    def test_decline_records_declined_status(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "pip install something"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(False, "declined", True)):
            update = apply_fix_node(base_state)
        finding = update["dep_findings"][0]
        assert finding.status == DepStatus.DECLINED

    def test_appends_install_output_to_terminal_log(self, base_state, tmp_path):
        """The install output MUST be in terminal_log_history for the next Groq call."""
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "pip install requests"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(True, "Successfully installed requests-2.31.0", False)):
            update = apply_fix_node(base_state)
        log = update["terminal_log_history"][0]
        assert "pip install requests" in log
        assert "Successfully installed requests-2.31.0" in log

    def test_skipped_to_report_false_on_success(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "pip install requests"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(True, "ok", False)):
            update = apply_fix_node(base_state)
        assert update["skipped_to_report"] is False


# ===========================================================================
# 9. report_node()
# ===========================================================================

class TestReportNode:

    def test_returns_final_report_path(self, base_state, tmp_path):
        expected = str(tmp_path / "SETUP_DIAGNOSTICS.md")
        with patch("contribtriage.graph.nodes.generate_report",
                   return_value=expected):
            update = report_node(base_state)
        assert update["final_report_path"] == expected

    def test_calls_generate_report_with_state(self, base_state, tmp_path):
        with patch("contribtriage.graph.nodes.generate_report",
                   return_value="/tmp/report.md") as mock_gen:
            report_node(base_state)
        mock_gen.assert_called_once_with(base_state)


# ===========================================================================
# 10. _infer_dep_type() — ecosystem label from fix_command
# ===========================================================================

class TestInferDepType:
    """Verify dep_type is correctly derived from the Groq fix command string."""

    def test_system_dep_always_returns_system(self):
        assert _infer_dep_type("system_dep", "docker run postgres") == "system"

    def test_system_dep_ignores_command(self):
        assert _infer_dep_type("system_dep", "") == "system"

    def test_npm_install_returns_node(self):
        assert _infer_dep_type("app_dep", "npm install express") == "node"

    def test_pnpm_add_returns_node(self):
        assert _infer_dep_type("app_dep", "pnpm add axios") == "node"

    def test_yarn_add_returns_node(self):
        assert _infer_dep_type("app_dep", "yarn add react") == "node"

    def test_npx_returns_node(self):
        assert _infer_dep_type("app_dep", "npx create-react-app .") == "node"

    def test_cargo_add_returns_rust(self):
        assert _infer_dep_type("app_dep", "cargo add serde") == "rust"

    def test_go_get_returns_go(self):
        assert _infer_dep_type("app_dep", "go get github.com/gin-gonic/gin") == "go"

    def test_go_mod_tidy_returns_go(self):
        assert _infer_dep_type("app_dep", "go mod tidy") == "go"

    def test_pip_install_returns_python(self):
        assert _infer_dep_type("app_dep", "pip install requests") == "python"

    def test_uv_install_returns_python(self):
        assert _infer_dep_type("app_dep", "uv pip install requests") == "python"

    def test_empty_command_defaults_to_python(self):
        assert _infer_dep_type("app_dep", "") == "python"


# ===========================================================================
# 11. apply_fix_node() — ecosystem labelling
# ===========================================================================

class TestApplyFixNodeEcosystem:
    """Verify DependencyFinding.dep_type matches the command ecosystem."""

    def test_npm_fix_labelled_node(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "npm install express"
        base_state["failure_category"] = "app_dep"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(True, "ok", False)):
            update = apply_fix_node(base_state)
        assert update["dep_findings"][0].dep_type == "node"

    def test_cargo_fix_labelled_rust(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "cargo add serde"
        base_state["failure_category"] = "app_dep"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(True, "ok", False)):
            update = apply_fix_node(base_state)
        assert update["dep_findings"][0].dep_type == "rust"

    def test_go_get_fix_labelled_go(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "go get github.com/gin-gonic/gin"
        base_state["failure_category"] = "app_dep"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(True, "ok", False)):
            update = apply_fix_node(base_state)
        assert update["dep_findings"][0].dep_type == "go"

    def test_pip_fix_labelled_python(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "pip install requests"
        base_state["failure_category"] = "app_dep"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(True, "ok", False)):
            update = apply_fix_node(base_state)
        assert update["dep_findings"][0].dep_type == "python"

    def test_docker_system_dep_labelled_system(self, base_state, tmp_path):
        base_state["repo_path"] = str(tmp_path)
        base_state["fix_command"] = "docker run -d postgres"
        base_state["failure_category"] = "system_dep"
        with patch("contribtriage.graph.nodes.run_fix_command",
                   return_value=(True, "ok", False)):
            update = apply_fix_node(base_state)
        assert update["dep_findings"][0].dep_type == "system"
