"""
contribtriage/cli.py

ContribTriage — Command-Line Entry Point.

Usage:
    contribtriage --repo-path ./path/to/cloned-repo [OPTIONS]

    Options:
        --repo-path   PATH    Path to the cloned repository (required)
        --yes                 Auto-accept all fix commands (non-interactive mode)
        --max-retries INT     Maximum heal-and-rerun cycles (default: 3)
        --persist             Enable LangGraph SqliteSaver checkpointing

    Environment variables:
        GROQ_API_KEY          Required — Groq API key for Llama 3 analysis
        GEMINI_API_KEY        Required — Gemini API key for report synthesis

    .env file:
        Loaded automatically from <repo-path>/.env or project root .env
        as a fallback when environment variables are not set.

What this file does:
    1. Loads API keys from env / .env
    2. Validates --repo-path exists
    3. Initialises the Groq client
    4. Builds the initial LangGraphState dict
    5. Calls build_graph().invoke(state)   ← LangGraph runs the full pipeline
    6. Prints the path to SETUP_DIAGNOSTICS.md

The entire pipeline — stages 1 through 6 — runs inside LangGraph's
StateGraph engine. This file is intentionally thin: its only job is
to wire CLI arguments into the initial state and hand control to
build_graph().
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()


# ===========================================================================
# Entry Point
# ===========================================================================

def main() -> None:
    """
    Parse CLI arguments, validate inputs, and invoke the LangGraph pipeline.
    """
    args = _parse_args()

    # ── Load API keys from .env fallback ──────────────────────────────────
    _load_env(args.repo_path)

    groq_key   = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    if not groq_key:
        console.print(
            "[red]✗  GROQ_API_KEY is not set.\n"
            "   Export it or add it to a .env file in the repo root.[/red]"
        )
        sys.exit(1)

    if not gemini_key:
        console.print(
            "[yellow]⚠  GEMINI_API_KEY is not set.\n"
            "   The final report will use a fallback template without Gemini.[/yellow]"
        )

    # ── Validate repo path ────────────────────────────────────────────────
    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        console.print(f"[red]✗  Repo path does not exist: {repo_path}[/red]")
        sys.exit(1)

    # ── Initialise Groq client ────────────────────────────────────────────
    try:
        from groq import Groq  # noqa: PLC0415
        groq_client = Groq(api_key=groq_key)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗  Could not initialise Groq client: {exc}[/red]")
        sys.exit(1)

    # ── Print banner ──────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold cyan]ContribTriage v2[/bold cyan]\n"
        f"[dim]Repo:[/dim] {repo_path}\n"
        f"[dim]Max retries:[/dim] {args.max_retries}  "
        f"[dim]Auto-accept:[/dim] {'yes' if args.yes else 'no'}  "
        f"[dim]Persist:[/dim] {'yes' if args.persist else 'no'}\n\n"
        f"[dim]LangGraph StateGraph engine starting…[/dim]",
        title="🚀  ContribTriage",
        border_style="cyan",
    ))

    # ── Build initial LangGraphState ──────────────────────────────────────
    initial_state = {
        # CLI inputs — never mutated by nodes
        "repo_path":   str(repo_path),
        "yes_flag":    args.yes,
        "max_retries": args.max_retries,
        "persist":     args.persist,

        # LLM clients injected once — passed through state
        "groq_client": groq_client,

        # Pipeline outputs — populated by their respective stage nodes
        "knowledge_graph":   None,   # set by ingest_node      (Stage 1)
        "vector_store_path": "",     # set by ingest_node      (Stage 1)
        "project_meta":      None,   # set by manifest_node    (Stage 2)
        "env_report":        None,   # set by audit_node       (Stage 3)
        "test_result":       None,   # set by run_tests_node   (Stage 4/5)
        "final_report_path": "",     # set by report_node      (Stage 6)

        # Accumulator fields — nodes append to these
        "terminal_log_history": [],
        "dep_findings":         [],

        # Healing loop guards — start at zero/empty
        "retry_count":       0,
        "skipped_to_report": False,
        "failure_category":  "",
        "failed_test_ids":   [],
        "fix_command":       "",
        "groq_analysis":     "",
    }

    # ── Compile and invoke the LangGraph StateGraph ───────────────────────
    try:
        from contribtriage.graph import build_graph  # noqa: PLC0415

        graph = build_graph()

        # Optional: SqliteSaver checkpointing for long runs
        if args.persist:
            _attach_checkpointer(graph, repo_path)

        final_state = graph.invoke(initial_state)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠  Interrupted by user.[/yellow]")
        sys.exit(130)

    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[red]✗  Pipeline error: {exc}[/red]")
        raise

    # ── Print result ──────────────────────────────────────────────────────
    report_path = final_state.get("final_report_path", "")
    if report_path:
        console.print(
            f"\n[bold green]✓  Done![/bold green]  "
            f"Report: [cyan]{report_path}[/cyan]"
        )
    else:
        console.print(
            "\n[yellow]⚠  Pipeline finished but no report path was set.[/yellow]"
        )


# ===========================================================================
# Argument Parser
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contribtriage",
        description=(
            "Automated open-source contributor onboarding: maps codebases, "
            "audits environments, resolves dependencies, and surfaces "
            "contribution pathways."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  contribtriage --repo-path ./httpie\n"
            "  contribtriage --repo-path ./next.js --yes --max-retries 5\n"
            "  contribtriage --repo-path ./serde --persist\n\n"
            "Environment variables:\n"
            "  GROQ_API_KEY     Groq API key (required)\n"
            "  GEMINI_API_KEY   Gemini API key (for full report synthesis)\n"
        ),
    )

    parser.add_argument(
        "--repo-path",
        required=True,
        metavar="PATH",
        help="Absolute or relative path to the cloned repository.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help=(
            "Auto-accept all proposed fix commands without prompting. "
            "Useful for CI pipelines."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        metavar="N",
        help="Maximum heal-and-rerun cycles before routing to report (default: 3).",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        default=False,
        help=(
            "Enable LangGraph SqliteSaver checkpointing. "
            "Saves state to .contribtriage_checkpoint.db so a long run can "
            "be resumed if interrupted."
        ),
    )

    return parser.parse_args()


# ===========================================================================
# Helpers
# ===========================================================================

def _load_env(repo_path: str) -> None:
    """
    Load .env files in priority order:
      1. <repo_path>/.env   (project-local, most specific)
      2. ./.env             (current working directory)
    Does NOT override already-set environment variables.
    """
    repo_env = Path(repo_path) / ".env"
    cwd_env  = Path(".env")

    for env_file in (repo_env, cwd_env):
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            break


def _attach_checkpointer(graph, repo_path: Path) -> None:
    """
    Attach a SqliteSaver checkpointer to the compiled graph for persistence.

    The checkpoint database is written to <repo_path>/.contribtriage_checkpoint.db
    so it stays local to the analysed repository.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415

        db_path = repo_path / ".contribtriage_checkpoint.db"
        checkpointer = SqliteSaver.from_conn_string(str(db_path))
        # SqliteSaver is attached at compile time — re-compile with it
        graph.__dict__["checkpointer"] = checkpointer
        console.print(
            f"[dim]  Checkpointing enabled → {db_path}[/dim]"
        )
    except ImportError:
        console.print(
            "[yellow]  ⚠ langgraph.checkpoint.sqlite not available — "
            "running without persistence.[/yellow]"
        )


if __name__ == "__main__":
    main()
