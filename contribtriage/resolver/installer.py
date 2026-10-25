"""
contribtriage/resolver/installer.py

Stage 5 — User Consent Gate + Command Executor.

This module is called by apply_fix_node in graph/nodes.py.
It is intentionally thin: it handles ONLY the consent prompt and subprocess
execution.  It does NOT decide WHAT to install — that is Groq's job.
It does NOT update LangGraph state — that is apply_fix_node's job.

The caller (apply_fix_node) is responsible for:
  1. Getting fix_command from state (set by analyze_failure_node)
  2. Calling run_fix_command()
  3. Appending the returned log to terminal_log_history
  4. Incrementing retry_count

Return contract
───────────────
  (success: bool, log: str, declined: bool)
  - success=True,  declined=False  → command ran and exited 0
  - success=False, declined=False  → command ran and exited non-0
  - success=False, declined=True   → user said N; no command was run
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from typing import Tuple

from rich.console import Console
from rich.panel import Panel


console = Console()

_TIMEOUT = 120  # seconds

# First word -> allowed sub-commands (anything else is refused)
_ALLOWED = {
    "pip": {"install"}, "pip3": {"install"},
    "uv": {"pip", "add", "sync"},
    "npm": {"install", "i", "ci"}, "pnpm": {"add", "install", "i"},
    "yarn": {"add", "install"},
    "cargo": {"add", "install", "fetch"},
    "go": {"get", "mod"},
    "docker": {"run", "compose"},
    "brew": {"install"}, "winget": {"install"}, "choco": {"install"},
}
_CHAIN_TOKENS = {"&&", "||", ";", "|", ">", ">>", "<", "&"}
_BAD_FLAGS = {
    "--index-url", "--extra-index-url", "--trusted-host", "--registry",
    "--privileged", "--cap-add", "--volume", "--mount",
}
_PIP_TOOLS = {"pip", "pip3", "uv", "python", "python3", "py"}


def _split(cmd: str) -> list:
    """Split a command string into words without using a shell."""
    if os.name == "nt":
        parts = shlex.split(cmd, posix=False)
        return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]
    return shlex.split(cmd)


def validate_fix_command(fix_command: str) -> Tuple[bool, str]:
    """Return (ok, reason). Only known install-type commands are allowed."""
    raw = fix_command.strip()
    if not raw:
        return False, "empty command"
    if "\n" in raw or "\r" in raw or "`" in raw or "$(" in raw:
        return False, "multi-line commands and shell substitution are not allowed"
    try:
        args = _split(raw)
    except ValueError as exc:
        return False, f"could not parse command ({exc})"
    if not args:
        return False, "empty command"
    if args[0] == "sudo":
        return False, "needs admin rights - please run it yourself in a terminal"
    if any(a in _CHAIN_TOKENS for a in args):
        return False, "command chaining or redirection is not allowed (one command only)"

    tool = args[0].lower()
    if tool in ("python", "python3", "py"):
        if args[1:4] != ["-m", "pip", "install"]:
            return False, "only `python -m pip install` is allowed"
    else:
        if tool not in _ALLOWED or len(args) < 2:
            return False, f"'{args[0]}' is not on the allowed list"
        sub, rest = args[1], args[2:]
        if sub not in _ALLOWED[tool]:
            return False, f"'{tool} {sub}' is not allowed"
        if tool == "uv" and sub == "pip" and (not rest or rest[0] != "install"):
            return False, "only `uv pip install` is allowed"
        if tool == "go" and sub == "mod" and (not rest or rest[0] not in ("download", "tidy")):
            return False, "only `go mod download|tidy` is allowed"
        if tool == "docker" and sub == "compose" and (not rest or rest[0] != "up"):
            return False, "only `docker compose up` is allowed"

    bad = set(_BAD_FLAGS)
    if tool in _PIP_TOOLS:
        bad.add("-i")
    if tool == "docker":
        bad.add("-v")
    for a in args[1:]:
        if a.split("=")[0] in bad:
            return False, f"flag '{a}' is not allowed"
        if "://" in a or a.startswith("git+"):
            return False, "installing from a URL is not allowed"
    return True, ""


# ===========================================================================
# Public API
# ===========================================================================

def run_fix_command(
    fix_command: str,
    yes_flag: bool,
    repo_root: str,
) -> Tuple[bool, str, bool]:
    """
    Present the LLM-generated fix command to the user and optionally execute it.

    Args:
        fix_command: The exact shell command string from Groq.
        yes_flag:    If True, execute without prompting (--yes CLI flag).
        repo_root:   Working directory for the subprocess.

    Returns:
        (success, log, declined)
    """
    if not fix_command or not fix_command.strip():
        return False, "[ContribTriage] No fix command provided.", False
    
    ok, reason = validate_fix_command(fix_command)
    if not ok:
        msg = f"[ContribTriage] Blocked command: {reason}\nCommand was: {fix_command}"
        console.print(f"[red]  ✗ {msg}[/red]")
        return False, msg, False

    _show_command_panel(fix_command)

    # ── Consent gate ──────────────────────────────────────────────────────
    if not _get_consent(fix_command, yes_flag):
        return False, "User declined the proposed fix.", True

    # ── Execute ───────────────────────────────────────────────────────────
    return _execute(fix_command, repo_root)


# ===========================================================================
# Consent prompt
# ===========================================================================

def _show_command_panel(fix_command: str) -> None:
    """Render a Rich panel showing the proposed fix."""
    console.print(Panel(
        f"[bold yellow]{fix_command}[/bold yellow]",
        title="🔧  Proposed Fix Command",
        subtitle="[dim]Generated by Groq Llama 3[/dim]",
        border_style="yellow",
    ))


def _get_consent(fix_command: str, yes_flag: bool) -> bool:
    """
    Ask the user whether to run the fix.
    Returns True if accepted, False if declined.
    """
    if yes_flag:
        console.print("[dim]  --yes flag set — auto-accepting.[/dim]")
        return True

    try:
        answer = input("  Run this command? [y/N]  ").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        # Non-interactive context (CI) — treat as declined
        console.print("[dim]  Non-interactive context — treating as declined.[/dim]")
        return False


# ===========================================================================
# Subprocess executor
# ===========================================================================

def _execute(fix_command: str, repo_root: str) -> Tuple[bool, str, bool]:
    """
    Run fix_command as a native shell subprocess.

    Uses shell=True because the LLM generates full shell strings including
    pipes, flags, and quoted arguments (e.g. `docker run -d -p 5432:5432 postgres:15`).

    Returns (success, log, declined=False).
    Never raises — all errors captured in log.
    """
    console.print(f"[cyan]  → Executing: {fix_command}[/cyan]")
    try:
        # proc = subprocess.run(
        #     fix_command,
        #     shell=True,
        #     cwd=repo_root,
        #     capture_output=True,
        #     text=True,
        #     timeout=_TIMEOUT,
        # )
        args = _split(fix_command)
        resolved = shutil.which(args[0])      # finds npm.cmd etc. on Windows
        if resolved:
            args[0] = resolved
        proc = subprocess.run(
            args,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
        log     = _combine(proc.stdout, proc.stderr)
        success = proc.returncode == 0

        icon = "✓" if success else "✗"
        color = "green" if success else "red"
        console.print(f"  [{color}]  {icon} Exit code {proc.returncode}[/{color}]")
        return success, log, False

    except subprocess.TimeoutExpired:
        msg = f"[ContribTriage] Timed out after {_TIMEOUT}s."
        console.print(f"[red]  ✗ {msg}[/red]")
        return False, msg, False

    except (OSError, Exception) as exc:  # noqa: BLE001
        msg = f"[ContribTriage] Could not execute command: {exc}"
        console.print(f"[red]  ✗ {msg}[/red]")
        return False, msg, False


def _combine(stdout: str, stderr: str) -> str:
    parts = [p.strip() for p in (stdout, stderr) if p and p.strip()]
    return "\n".join(parts)
