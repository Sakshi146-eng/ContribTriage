"""
contribtriage/graph/nodes.py

LangGraph Node Functions — the Execution States.

Every function here is a LangGraph node:
  - Takes the full LangGraphState as input
  - Returns a PARTIAL state update dict (LangGraph merges it in)
  - Side effects are allowed (subprocess, file I/O, API calls)
  - terminal_log_history updates use the append reducer automatically

State accumulation contract
───────────────────────────
  Every node that runs a subprocess MUST append its output to
  terminal_log_history by including it in the returned dict.
  LangGraph's Annotated[List[str], operator.add] reducer concatenates
  the returned list onto the existing one — no node ever overwrites
  another node's logs.

  Format convention for log entries:
    "[STAGE N — <NodeName> #<retry>]\n<content>"

Node inventory
──────────────
  ingest_node         → Stage 1: lexical parser + vector store
  manifest_node       → Stage 2: manifest parsing → project_meta
  audit_node          → Stage 3: env audit → env_report
  check_coverage_node → Stage 4a: KG vs test file coverage check
  generate_tests_node → Stage 4b: write dep-import stubs
  run_tests_node      → Stage 4c/5b: execute test suite (full or selective)
  analyze_failure_node→ Stage 5a: send full history to Groq → classify
  apply_fix_node      → Stage 5c: user consent + run fix + append log
  report_node         → Stage 6: Gemini synthesises SETUP_DIAGNOSTICS.md
"""

from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console

from contribtriage.models import (
    DependencyFinding,
    DepStatus,
    LangGraphState,
    TestResult,
)

# Service imports — at module level so unittest.mock.patch can intercept them
from contribtriage.runners.test_runner import run_tests
from contribtriage.runners.test_generator import generate_module_test_files
from contribtriage.audit.env_auditor import audit_environment
from contribtriage.ingestion.doc_reader import parse_manifests
from contribtriage.agents.groq_agent import analyze_failure
from contribtriage.resolver.installer import run_fix_command
from contribtriage.report.report_generator import generate_report

console = Console()


# ===========================================================================
# Stage 1 — Ingestion
# ===========================================================================

def ingest_node(state: LangGraphState) -> Dict[str, Any]:
    """
    Run the Universal Lexical Parser and populate the Qdrant vector store.

    Reads all source files in repo_path, extracts modules/functions/imports
    into a KnowledgeGraph, and ingests unstructured docs (README, CONTRIBUTING)
    into Qdrant using FastEmbed.

    Note: lexical_parser and vector_store are imported lazily here because
    they optionally load Qdrant/FastEmbed — heavy optional deps that may
    not be installed in test environments.
    """
    from contribtriage.ingestion.lexical_parser import build_knowledge_graph  # noqa
    from contribtriage.ingestion.vector_store import VectorStore              # noqa

    repo_path = state["repo_path"]
    console.print(f"[cyan]  [Stage 1] Parsing codebase: {repo_path}[/cyan]")

    kg    = build_knowledge_graph(repo_path)
    store = VectorStore.from_repo(repo_root=repo_path)
    n_chunks = store.ingest_repo(kg, repo_path)

    n_modules = len(getattr(kg, "nodes", {}))
    console.print(f"  [green]✓[/green] KG: {n_modules} module(s) indexed")

    log = (
        f"[Stage 1 — Ingestion]\n"
        f"Repo: {repo_path}\n"
        f"Modules indexed: {n_modules}\n"
        f"Vector store chunks: {n_chunks}"
    )
    return {
        "knowledge_graph":      kg,
        "vector_store_path":    store._path,
        "terminal_log_history": [log],
    }


# ===========================================================================
# Stage 2 — Manifest Parsing
# ===========================================================================

def manifest_node(state: LangGraphState) -> Dict[str, Any]:
    """
    Parse all manifest files (pyproject.toml, package.json, Cargo.toml, go.mod)
    into a unified ProjectMeta object.
    """
    repo_path = state["repo_path"]
    console.print("[cyan]  [Stage 2] Parsing project manifests…[/cyan]")

    meta = parse_manifests(repo_path)
    n_deps = len(getattr(meta, "declared_deps", []))
    console.print(f"  [green]✓[/green] {n_deps} declared dep(s) found")

    log = (
        f"[Stage 2 — Manifest Parsing]\n"
        f"Ecosystem: {getattr(meta, 'ecosystem', 'unknown')}\n"
        f"Declared deps ({n_deps}): "
        f"{', '.join(str(d) for d in getattr(meta, 'declared_deps', [])[:20])}\n"
        f"Test framework: {getattr(meta, 'test_framework', 'unknown')}"
    )
    return {
        "project_meta": meta,
        "terminal_log_history": [log],
    }


# ===========================================================================
# Stage 3 — Environment Audit
# ===========================================================================

def audit_node(state: LangGraphState) -> Dict[str, Any]:
    """
    Audit the host environment: OS, Python/Node/Rust/Go versions, venv,
    package managers, Docker daemon, system tools.
    """
    repo_path = state["repo_path"]
    console.print("[cyan]  [Stage 3] Auditing host environment…[/cyan]")

    env = audit_environment(repo_path)
    console.print(
        f"  [green]✓[/green] {env.os_name} | Python {env.python_version} | "
        f"Docker {'✓' if env.docker_running else '✗'}"
    )

    log = (
        f"[Stage 3 — Environment Audit]\n"
        f"OS: {env.os_name} {env.os_version}\n"
        f"Python: {env.python_version} ({env.python_path})\n"
        f"Node: {env.node_version or 'not found'}\n"
        f"Rust: {env.rust_version or 'not found'}\n"
        f"Go: {env.go_version or 'not found'}\n"
        f"Docker: {'running' if env.docker_running else 'not running'}\n"
        f"venv: {env.venv_type or 'none'} | "
        f"uv: {env.uv_available} | pip: {env.pip_available} | "
        f"npm: {env.npm_available}"
    )
    return {
        "env_report": env,
        "terminal_log_history": [log],
    }


# ===========================================================================
# Stage 4a — Coverage Check
# ===========================================================================

def check_coverage_node(state: LangGraphState) -> Dict[str, Any]:
    """
    Read uncovered public functions from the KnowledgeGraph.

    Tree-sitter's build_knowledge_graph already computes kg.uncovered_funcs
    correctly for ALL languages (Python, JS, Go, Rust) during Stage 1.
    This node simply reads that result and logs it — no redundant scan needed.
    """
    kg           = state.get("knowledge_graph")
    project_meta = state.get("project_meta")
    declared_deps = getattr(project_meta, "declared_deps", []) if project_meta else []

    uncovered_funcs = getattr(kg, "uncovered_funcs", []) if kg else []

    log = (
        f"[Stage 4a — Coverage Check]\n"
        f"Uncovered KG functions: {len(uncovered_funcs)} "
        f"(computed by Tree-sitter lexical parser)\n"
        f"Declared deps in manifest: {len(declared_deps)}"
    )
    console.print(
        f"  [dim]Coverage: {len(uncovered_funcs)} uncovered functions "
        f"across {len(getattr(kg, 'nodes', {})) } modules[/dim]"
    )
    return {
        "knowledge_graph":       kg,
        "terminal_log_history":  [log],
    }


# ===========================================================================
# Stage 4b — Test Stub Generation
# ===========================================================================

def generate_tests_node(state: LangGraphState) -> Dict[str, Any]:
    """
    Generate Groq-powered test files — one per source module.

    Each generated file contains:
      - Import block: imports the source module + its deps (dep health check)
      - Test stubs: one stub per uncovered public function

    Groq (Llama-3.3-70b) generates each file dynamically in the correct
    language/framework format. Syntax is validated before writing; invalid
    AI output falls back to a minimal valid template.
    """
    repo_path    = state.get("repo_path", ".")
    kg           = state.get("knowledge_graph")
    project_meta = state.get("project_meta")
    groq_client  = state.get("groq_client")

    generated_paths = generate_module_test_files(
        knowledge_graph=kg,
        project_meta=project_meta,
        groq_client=groq_client,
        repo_root=repo_path,
    )

    console.print(
        f"  [green]✓[/green] Generated {len(generated_paths)} test file(s) "
        f"(imports + stubs, AI-powered)"
    )

    log = (
        f"[Stage 4b — AI Test Stub Generation]\n"
        f"Files generated: {len(generated_paths)}\n"
        f"Paths: {', '.join(generated_paths[:10])}"
    )
    return {"terminal_log_history": [log]}


# ===========================================================================
# Stage 4c / 5b — Test Runner (full + selective rerun)
# ===========================================================================

def run_tests_node(state: LangGraphState) -> Dict[str, Any]:
    """
    Execute the test suite and append the full output to terminal_log_history.

    Selective rerun: if failed_test_ids is populated from a previous run,
    the runner executes ONLY those tests (not the full suite) for speed.

    The raw output MUST be appended to terminal_log_history so the next
    analyze_failure_node call can see it alongside all previous history.
    """
    repo_path    = state.get("repo_path", ".")
    project_meta = state.get("project_meta")
    retry_count  = state.get("retry_count", 0)
    failed_ids   = state.get("failed_test_ids", [])

    # test_framework is List[str] — pick the first valid entry, fall back to pytest
    raw_frameworks = getattr(project_meta, "test_framework", []) if project_meta else []
    framework = next(
        (f for f in raw_frameworks if f and f != "(none detected)"),
        "pytest",   # safe default for any unrecognised repo
    )

    run_label = (
        f"Selective rerun #{retry_count} ({len(failed_ids)} test(s))"
        if failed_ids
        else "Full test suite"
    )
    console.print(f"[cyan]  [Stage 4/5] {run_label} [{framework}]…[/cyan]")

    result = run_tests(repo_path, framework, failed_test_ids=failed_ids)

    # Format for terminal_log_history — Groq will read this verbatim
    log = (
        f"[Test Run #{retry_count + 1} — {run_label}]\n"
        f"Framework: {framework}\n"
        f"Command: {' '.join(result.command_used)}\n"
        f"Results: {result.passed} passed / {result.failed} failed / "
        f"{result.errors} errors / {result.skipped} skipped\n"
        f"--- stdout/stderr ---\n{result.raw_output}"
    )

    new_failed_ids = [item[0] for item in result.failed_items]

    status = "✓" if result.failed == 0 and result.errors == 0 else "✗"
    color  = "green" if result.failed == 0 and result.errors == 0 else "red"
    console.print(
        f"  [{color}]{status}[/{color}] {result.passed} passed / "
        f"{result.failed} failed"
    )

    return {
        "test_result":         result,
        "failed_test_ids":     new_failed_ids,
        "terminal_log_history": [log],
    }


# ===========================================================================
# Stage 5a — Failure Analysis (Groq)
# ===========================================================================

def analyze_failure_node(state: LangGraphState) -> Dict[str, Any]:
    """
    Merge the FULL terminal_log_history and send it to Groq for analysis.

    This is the non-blind loop design:
    - First call: Groq sees the original test failure
    - Second call: Groq sees test failure + previous install attempt
    - Third call: Groq sees all of the above + rerun result

    Groq cannot be fooled into repeating the same failed command because
    it can see the prior attempt's output in the accumulated history.
    """
    groq_client  = state.get("groq_client")
    env_report   = state.get("env_report")
    project_meta = state.get("project_meta")
    failed_ids   = state.get("failed_test_ids", [])

    # Merge ALL accumulated history into one context block for Groq
    history      = state.get("terminal_log_history", [])
    full_context = "\n\n".join(history)

    console.print(
        f"[cyan]  [Stage 5a] Groq analysing "
        f"({len(history)} log entries, {len(full_context)} chars)…[/cyan]"
    )

    analysis = analyze_failure(
        raw_output=full_context,     # THE FULL HISTORY — not just latest run
        env_report=env_report,
        project_meta=project_meta,
        groq_client=groq_client,
        failed_test_ids=failed_ids,
    )

    log = (
        f"[Stage 5a — Groq Analysis]\n"
        f"Category: {analysis.get('failure_category', 'unknown')}\n"
        f"Reasoning: {analysis.get('reasoning', '')}\n"
        f"Fix command: {analysis.get('fix_command', 'N/A')}"
    )

    return {
        "failure_category":   analysis.get("failure_category", "code_bug"),
        "groq_analysis":      analysis.get("reasoning", ""),
        "fix_command":        analysis.get("fix_command") or "",
        "terminal_log_history": [log],
    }


# ===========================================================================
# Stage 5c — Apply Fix (user consent + execute + accumulate)
# ===========================================================================

def apply_fix_node(state: LangGraphState) -> Dict[str, Any]:
    """
    Present the LLM-generated fix command to the user, execute it on consent,
    and append the full install output to terminal_log_history.

    State accumulation is the core responsibility of this node:
    The install output is appended so the NEXT analyze_failure_node call
    can see exactly what the installer printed — success or failure.
    """
    fix_command  = state.get("fix_command", "")
    yes_flag     = state.get("yes_flag", False)
    repo_root    = state.get("repo_path", ".")
    retry_count  = state.get("retry_count", 0)
    category     = state.get("failure_category", "")

    console.print(
        f"[cyan]  [Stage 5c] Applying fix (attempt #{retry_count + 1})…[/cyan]"
    )

    success, install_log, declined = run_fix_command(fix_command, yes_flag, repo_root)

    # Format log entry — Groq reads this on the next analyze_failure call
    status_label = "DECLINED" if declined else ("SUCCESS" if success else "FAILED")
    log = (
        f"[Stage 5c — Fix Attempt #{retry_count + 1}]\n"
        f"Command: {fix_command}\n"
        f"Status: {status_label}\n"
        f"Output:\n{install_log}"
    )

    dep_type = _infer_dep_type(category, fix_command)
    new_finding = DependencyFinding(
        name=fix_command[:80],
        dep_type=dep_type,
        status=(
            DepStatus.DECLINED  if declined else
            DepStatus.INSTALLED if success  else
            DepStatus.FAILED
        ),
        install_log=install_log,
        notes="" if success else f"Manual fallback: {fix_command}",
    )

    return {
        "terminal_log_history": [log],
        "retry_count":          retry_count + 1,
        "dep_findings":         [new_finding],
        "skipped_to_report":    declined,
    }


# ===========================================================================
# Stage 6 — Report Generation (Gemini)
# ===========================================================================

def report_node(state: LangGraphState) -> Dict[str, Any]:
    """
    Call Gemini 2.5 Flash with the full accumulated state to synthesise
    SETUP_DIAGNOSTICS.md.  Write the file to the repo root.
    """
    console.print("[cyan]  [Stage 6] Generating SETUP_DIAGNOSTICS.md…[/cyan]")
    path = generate_report(state)
    return {"final_report_path": path}


# ===========================================================================
# Private helpers
# ===========================================================================

def _infer_dep_type(category: str, fix_command: str) -> str:
    """
    Infer the correct dep_type label for a DependencyFinding from the
    failure category and the actual fix command Groq generated.

    Why this matters: Groq may generate `npm install express` for a Node
    repo — labelling that as "python" in the report is wrong and confusing.

    Rules (checked in order against the first token of fix_command):
      system_dep           → "system"   (docker run, apt install, brew, winget)
      npm/pnpm/yarn        → "node"
      cargo                → "rust"
      go                   → "go"
      pip/uv/pipx/poetry   → "python"
      anything else        → "python"   (safe default)
    """
    if category == "system_dep":
        return "system"

    cmd = fix_command.strip().lower() if fix_command else ""
    first_token = cmd.split()[0] if cmd.split() else ""

    if first_token in ("npm", "pnpm", "yarn", "npx"):
        return "node"
    if first_token == "cargo":
        return "rust"
    if first_token == "go":
        return "go"
    # pip, uv, pipx, poetry, pip3, python -m pip, …
    return "python"
