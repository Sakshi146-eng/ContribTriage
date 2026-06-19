"""
contribtriage/graph/edges.py

LangGraph Conditional Edge Routing Functions.

Every function in this module is a pure routing function:
  - Takes LangGraphState as input
  - Returns a string matching a node name
  - Has NO side effects — never mutates state, never runs subprocesses

These functions are the "decision diamonds" in the architecture diagram.
LangGraph calls them after each node to decide where to go next.

Routing map
───────────
  route_coverage    → "run_tests" | "generate_tests"
  route_test_result → "analyze_failure" | "report"
  route_failure     → "apply_fix" | "report"
  route_consent     → "run_tests" | "report"
"""

from contribtriage.models import LangGraphState


# ===========================================================================
# Edge: check_coverage_node → run_tests | generate_tests
# ===========================================================================

def route_coverage(state: LangGraphState) -> str:
    """
    Decide whether to run existing tests or generate new stubs first.

    Rules:
    - If the KG has uncovered functions OR there are declared deps
      without corresponding import-validation tests → generate stubs.
    - Otherwise → run tests directly.
    """
    kg           = state.get("knowledge_graph")
    project_meta = state.get("project_meta")
    declared     = getattr(project_meta, "declared_deps", []) if project_meta else []

    if kg is None:
        # No KG yet → can't assess coverage, just run what's there
        return "run_tests"

    uncovered_funcs = getattr(kg, "uncovered_funcs", [])
    if uncovered_funcs or declared:
        return "generate_tests"

    return "run_tests"


# ===========================================================================
# Edge: run_tests_node → analyze_failure | report
# ===========================================================================

def route_test_result(state: LangGraphState) -> str:
    """
    Decide what to do after a test run.

    - All tests passed AND no errors → route to report (done!)
    - Any failure or error → route to analyze_failure for LLM diagnosis
    """
    result = state.get("test_result")

    if result is None:
        # No result captured → treat as failure to be safe
        return "analyze_failure"

    if result.failed == 0 and result.errors == 0:
        return "report"

    return "analyze_failure"


# ===========================================================================
# Edge: analyze_failure_node → apply_fix | report
# ===========================================================================

def route_failure(state: LangGraphState) -> str:
    """
    Route based on Groq's failure classification.

    - code_bug  → do NOT attempt env fixes; route straight to report.
                  The report will surface these as contribution opportunities.
    - system_dep|app_dep → route to apply_fix_node.
    - Unknown   → treat as code_bug (safe default).
    """
    category = state.get("failure_category", "code_bug")

    if category in ("system_dep", "app_dep"):
        return "apply_fix"

    # code_bug or unrecognised → go to report
    return "report"


# ===========================================================================
# Edge: apply_fix_node → run_tests | report
# ===========================================================================

def route_consent(state: LangGraphState) -> str:
    """
    Route after apply_fix_node runs.

    Three exit conditions:
    1. User said N (skipped_to_report=True)  → report immediately
    2. Max retries exhausted                 → report immediately
    3. Fix was applied (success or fail)     → rerun tests to verify

    Note: even if the fix command FAILED, we still rerun tests.
    The next analyze_failure_node will see the failure in terminal_log_history
    and can suggest a different approach — that's the non-blind loop.
    """
    if state.get("skipped_to_report", False):
        return "report"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count >= max_retries:
        return "report"

    return "run_tests"
