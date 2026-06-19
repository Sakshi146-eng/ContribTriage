"""
contribtriage/graph/__init__.py

LangGraph StateGraph — The Central Pipeline Engine.

Compiles the full ContribTriage pipeline as a directed StateGraph.
Call build_graph() to get a compiled, invocable graph.

Graph topology
──────────────
  ingest → manifest → audit → check_coverage
                                     │
                          ┌──────────┴──────────┐
                     (has gaps)             (no gaps)
                          ▼                     ▼
                   generate_tests           run_tests ◄──────────────┐
                          │                     │                    │
                          └──────────┬──────────┘                    │
                                     ▼                               │
                                 run_tests                           │
                                     │                               │
                          ┌──────────┴──────────┐                   │
                      (passed)               (failed)               │
                          ▼                     ▼                   │
                        report          analyze_failure             │
                                               │                   │
                                  ┌────────────┴──────────┐        │
                              (code_bug)           (dep/sys)        │
                                  ▼                     ▼          │
                                report            apply_fix        │
                                                       │           │
                                              ┌────────┴────────┐  │
                                          (declined/           (Y+retry<max)
                                           max_retries)              │
                                               ▼                    └──┘
                                             report
"""

from langgraph.graph import StateGraph, END

from contribtriage.models import LangGraphState
from contribtriage.graph.nodes import (
    ingest_node,
    manifest_node,
    audit_node,
    check_coverage_node,
    generate_tests_node,
    run_tests_node,
    analyze_failure_node,
    apply_fix_node,
    report_node,
)
from contribtriage.graph.edges import (
    route_coverage,
    route_test_result,
    route_failure,
    route_consent,
)


def build_graph():
    """
    Compile and return the ContribTriage LangGraph StateGraph.

    Returns a compiled graph that can be invoked with an initial state dict:
        graph = build_graph()
        result = graph.invoke(initial_state)
    """
    g = StateGraph(LangGraphState)

    # ── Register nodes ─────────────────────────────────────────────────────
    g.add_node("ingest",          ingest_node)
    g.add_node("manifest",        manifest_node)
    g.add_node("audit",           audit_node)
    g.add_node("check_coverage",  check_coverage_node)
    g.add_node("generate_tests",  generate_tests_node)
    g.add_node("run_tests",       run_tests_node)
    g.add_node("analyze_failure", analyze_failure_node)
    g.add_node("apply_fix",       apply_fix_node)
    g.add_node("report",          report_node)

    # ── Set entry point ────────────────────────────────────────────────────
    g.set_entry_point("ingest")

    # ── Linear pipeline edges ──────────────────────────────────────────────
    g.add_edge("ingest",   "manifest")
    g.add_edge("manifest", "audit")
    g.add_edge("audit",    "check_coverage")

    # ── Conditional: coverage check → generate stubs or run directly ───────
    g.add_conditional_edges(
        "check_coverage",
        route_coverage,
        {"generate_tests": "generate_tests", "run_tests": "run_tests"},
    )
    g.add_edge("generate_tests", "run_tests")

    # ── Conditional: test result → analyze or report ───────────────────────
    g.add_conditional_edges(
        "run_tests",
        route_test_result,
        {"analyze_failure": "analyze_failure", "report": "report"},
    )

    # ── Conditional: failure category → fix or report ──────────────────────
    g.add_conditional_edges(
        "analyze_failure",
        route_failure,
        {"apply_fix": "apply_fix", "report": "report"},
    )

    # ── Conditional: consent result → rerun tests or report ────────────────
    g.add_conditional_edges(
        "apply_fix",
        route_consent,
        {"run_tests": "run_tests", "report": "report"},
    )

    # ── Terminal node ──────────────────────────────────────────────────────
    g.add_edge("report", END)

    return g.compile()


__all__ = ["build_graph"]
