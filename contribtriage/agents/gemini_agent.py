"""
contribtriage/agents/gemini_agent.py

Gemini 2.5 Flash — Synthesis Brain (Stage 6 only).

Role in the pipeline
────────────────────
  Called ONCE at the very end (report_node) after the healing loop exits.
  Gemini is chosen specifically for this job because of its massive context
  window — it can ingest the ENTIRE accumulated terminal_log_history
  (every test run + every install attempt) in a single API call.

  Groq handles fast, iterative decisions inside the healing loop.
  Gemini handles deep, comprehensive synthesis after the loop.

What it generates
─────────────────
  SETUP_DIAGNOSTICS.md containing:
    1. Environment Snapshot (OS, runtimes, venv, package managers)
    2. Codebase Map (language breakdown, KG stats, module list)
    3. Dependency Resolution Log (installed / failed / declined)
    4. Test Results Summary (passed / failed / generated stubs)
    5. Three Contribution Pathways:
         a. Failing tests → filed as reproducible bug reports
         b. TODO/FIXME/BUG sweep from lexical scanner
         c. Uncovered functions from KG → suggested test PRs

Design rules
────────────
  - Model: gemini-2.5-flash (large context window, strong synthesis)
  - Uses google-genai SDK (google.genai.Client)
  - API key: GEMINI_API_KEY env var
  - Never raises — errors produce a minimal fallback report.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from contribtriage.models import LangGraphState

console = Console()

_MODEL = "gemini-2.5-flash"


# ===========================================================================
# Public API
# ===========================================================================

def synthesize_report(state: "LangGraphState") -> str:
    """
    Call Gemini 2.5 Flash with the full accumulated session state and
    return the complete SETUP_DIAGNOSTICS.md content as a string.

    Args:
        state: The fully-populated LangGraphState after all nodes have run.

    Returns:
        Markdown string — the complete SETUP_DIAGNOSTICS.md content.
    """
    prompt = _build_report_prompt(state)

    console.print("[cyan]  → Gemini 2.5 Flash synthesising SETUP_DIAGNOSTICS.md…[/cyan]")

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        console.print("[yellow]  ⚠ GEMINI_API_KEY not set — generating fallback report[/yellow]")
        return _fallback_report(state)

    try:
        from google import genai as google_genai  # noqa: PLC0415

        client   = google_genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_MODEL,
            contents=prompt,
        )
        content = response.text
        console.print("[green]  ✓ SETUP_DIAGNOSTICS.md synthesised by Gemini[/green]")
        return content

    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]  ⚠ Gemini call failed ({exc}) — using fallback report[/yellow]")
        return _fallback_report(state)


# ===========================================================================
# Prompt builder
# ===========================================================================

def _build_report_prompt(state: "LangGraphState") -> str:
    """Build the synthesis prompt from accumulated state."""

    env_report   = state.get("env_report")
    project_meta = state.get("project_meta")
    test_result  = state.get("test_result")
    dep_findings = state.get("dep_findings", [])
    kg           = state.get("knowledge_graph")
    history      = state.get("terminal_log_history", [])
    groq_notes   = state.get("groq_analysis", "")
    failure_cat  = state.get("failure_category", "")

    # Environment section
    env_text = (
        f"OS: {getattr(env_report, 'os_name', 'unknown')} {getattr(env_report, 'os_version', '')}\n"
        f"Python: {getattr(env_report, 'python_version', 'unknown')}\n"
        f"Node: {getattr(env_report, 'node_version', 'N/A')}\n"
        f"Rust: {getattr(env_report, 'rust_version', 'N/A')}\n"
        f"Go: {getattr(env_report, 'go_version', 'N/A')}\n"
        f"Docker: {'running' if getattr(env_report, 'docker_running', False) else 'not running'}\n"
        f"venv: {getattr(env_report, 'venv_type', 'none')} ({getattr(env_report, 'venv_path', '')})"
    ) if env_report else "Environment data not available."

    # Dependency findings
    dep_text = "\n".join(
        f"  - {f.name} [{getattr(f, 'dep_type', '')}]: {getattr(f, 'status', {}).value if hasattr(getattr(f, 'status', None), 'value') else f.status}"
        for f in dep_findings
    ) if dep_findings else "  No dependency actions taken."

    # KG summary
    kg_text = (
        f"Modules: {len(getattr(kg, 'modules', {}))}\n"
        f"Functions: {sum(len(m.functions) for m in getattr(kg, 'modules', {}).values() if hasattr(m, 'functions'))}\n"
        f"Languages: {', '.join(set(getattr(m, 'language', 'python') for m in getattr(kg, 'modules', {}).values()))}"
    ) if kg else "Knowledge graph not available."

    # Test summary
    test_text = (
        f"Passed: {test_result.passed}\n"
        f"Failed: {test_result.failed}\n"
        f"Errors: {test_result.errors}\n"
        f"Skipped: {test_result.skipped}"
    ) if test_result else "Test results not available."

    # Full history — Gemini's large context window handles this
    history_text = "\n\n".join(history) if history else "No execution history recorded."

    return f"""\
You are generating a professional developer onboarding document called SETUP_DIAGNOSTICS.md.
A developer just cloned an open-source repository. ContribTriage ran a full automated
environment audit, test execution, and dependency healing session. Below is all the data
collected. Generate the complete markdown document now.

=== ENVIRONMENT SNAPSHOT ===
{env_text}

=== KNOWLEDGE GRAPH ===
{kg_text}

=== DEPENDENCY RESOLUTION LOG ===
{dep_text}

=== TEST RESULTS ===
{test_text}

=== GROQ DIAGNOSIS (final) ===
Category: {failure_cat}
Reasoning: {groq_notes}

=== FULL SESSION EXECUTION HISTORY ===
{history_text}

---

Generate SETUP_DIAGNOSTICS.md with these EXACT sections:
1. **Environment Snapshot** — table of OS/runtimes/tools/venv
2. **Codebase Map** — language breakdown, module count, function count from KG
3. **Dependency Resolution Log** — table of installed/failed/declined packages with commands
4. **Test Results** — passed/failed counts, exact commands to reproduce failures
5. **Contribution Pathways** — THREE pathways:

   ### 🐛 Pathway A: Code Bug Fix PRs
   For EACH failing test where Groq's diagnosis is "code_bug":
   - Quote the EXACT test ID and the full traceback from the session history
   - Identify the specific source file and line number that is wrong
   - Explain WHY it fails (logic error, assertion mismatch, type error, etc.)
   - Suggest the CORRECTED code as a fenced diff block (```diff ... ```)
   - Write a ready-to-use PR title and description the developer can open immediately
   - Label this: [ ] Good First Issue / Bug Fix

   ### 📝 Pathway B: Inline Annotation TODOs
   - List every TODO/FIXME/BUG/HACK/XXX comment found by the lexical scanner
   - Include file path, line number, and the annotation text
   - Suggest a PR title for each

   ### 🧪 Pathway C: Test Coverage PRs
   - List uncovered public functions from the KG cross-reference
   - For each, write the skeleton test function the contributor should add
   - Label this: [ ] Good First Issue / Test Coverage

Format as clean, copy-pasteable GitHub Markdown. Use tables, code blocks, and checkboxes.
Be specific — include actual test IDs, actual function names, actual file paths from the data.
For Pathway A, ALWAYS include the corrected code suggestion even if it requires reasoning from the traceback.
If failure_category is NOT code_bug, Pathway A should say: No code-logic failures detected - all failures were environment issues (handled above)."""


# ===========================================================================
# Fallback (no API key or Gemini call failed)
# ===========================================================================

def _fallback_report(state: "LangGraphState") -> str:
    """Generate a minimal but still useful report without Gemini."""
    ts         = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    env_report = state.get("env_report")
    test_result= state.get("test_result")
    history    = state.get("terminal_log_history", [])

    lines = [
        "# SETUP_DIAGNOSTICS.md",
        f"> Generated by ContribTriage on {ts} (fallback mode — Gemini unavailable)",
        "",
        "## Environment Snapshot",
        f"- OS: {getattr(env_report, 'os_name', 'unknown')}",
        f"- Python: {getattr(env_report, 'python_version', 'unknown')}",
        f"- Docker: {'running' if getattr(env_report, 'docker_running', False) else 'not running'}",
        "",
        "## Test Results",
        f"- Passed: {getattr(test_result, 'passed', 'N/A')}",
        f"- Failed: {getattr(test_result, 'failed', 'N/A')}",
        "",
        "## Full Execution Log",
        "```",
        "\n".join(history[-5:]) if history else "(no history)",
        "```",
        "",
        "> ⚠ Set GEMINI_API_KEY for a full synthesised report.",
    ]
    return "\n".join(lines)
