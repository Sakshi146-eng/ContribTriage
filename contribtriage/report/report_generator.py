"""
contribtriage/report/report_generator.py

Stage 6 — SETUP_DIAGNOSTICS.md Generator.

Called by report_node in graph/nodes.py after the healing loop exits.
Delegates all text synthesis to gemini_agent.synthesize_report(),
then writes the result to disk in the target repository.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

from contribtriage.agents.gemini_agent import synthesize_report

if TYPE_CHECKING:
    from contribtriage.models import LangGraphState

console = Console()

_FILENAME = "SETUP_DIAGNOSTICS.md"


# ===========================================================================
# Public API
# ===========================================================================

def generate_report(state: "LangGraphState") -> str:
    """
    Synthesise and write SETUP_DIAGNOSTICS.md.

    Calls Gemini 2.5 Flash with the complete accumulated state,
    writes the result to <repo_root>/SETUP_DIAGNOSTICS.md, and
    returns the absolute path for the state update.

    Args:
        state: Fully-populated LangGraphState after all nodes have run.

    Returns:
        Absolute path to the written SETUP_DIAGNOSTICS.md file.
    """
    console.print(Panel(
        "[bold]Synthesising contributor onboarding report…[/bold]\n"
        "[dim]Ingesting full session history into Gemini 2.5 Flash[/dim]",
        title="📄  Stage 6 — Report Generation",
        border_style="blue",
    ))

    # Gemini synthesises the full content
    content = synthesize_report(state)

    # Write to repo root (visible to the developer who cloned the repo)
    repo_root  = state.get("repo_path", ".")
    output_dir = Path(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / _FILENAME
    out_path.write_text(content, encoding="utf-8")

    console.print(Panel(
        f"[bold green]Report written to:[/bold green]\n"
        f"[cyan]{out_path}[/cyan]\n\n"
        f"[dim]Share this file with new contributors or open it in your editor.[/dim]",
        title="✅  ContribTriage Complete",
        border_style="green",
    ))

    return str(out_path.resolve())
