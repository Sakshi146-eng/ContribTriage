"""
contribtriage/audit/env_auditor.py

Stage 3 — Environment Audit.

Snapshots the contributor's local host in one pass:

  OS layer:
    - Platform name (Windows / Linux / Darwin) and version string
  
  Python layer:
    - Python version and interpreter path
    - Active virtual environment type (venv / conda / none) and path
  
  Package managers:
    - uv, pip           (Python)
    - npm, pnpm, yarn   (Node.js)
    - cargo             (Rust)
    - go                (Go)
  
  Language runtimes (versions via subprocess):
    - Node.js, Rust (rustc), Go
  
  Docker:
    - Whether the Docker daemon is reachable via `docker info`
  
  System tools:
    - postgres, redis, mongodb, mysql — detected via tool_checker

Design:
  - NEVER raises — every detection is wrapped in try/except.
  - Uses shutil.which for binary detection (no PATH assumptions).
  - subprocess calls use a 5-second timeout so audits stay fast.
  - Returns a fully populated EnvReport dataclass.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Optional, Tuple

from rich.console import Console

from contribtriage.audit.tool_checker import check_system_tools
from contribtriage.models import EnvReport

console = Console()


# ===========================================================================
# Public API
# ===========================================================================

def audit_environment(repo_root: str) -> EnvReport:
    """
    Snapshot the contributor's local environment.

    Args:
        repo_root: Absolute path to the analysed repository.
                   Currently unused but kept for future per-repo config lookup.

    Returns:
        Fully populated EnvReport. Any field that could not be detected is
        left at its dataclass default (None / False / "").
    """
    report = EnvReport()

    # ── OS ────────────────────────────────────────────────────────────────
    report.os_name    = platform.system()   # 'Windows', 'Linux', 'Darwin'
    report.os_version = platform.version()  # full version string

    # ── Python runtime ────────────────────────────────────────────────────
    report.python_version = platform.python_version()  # e.g. '3.11.4'
    report.python_path    = sys.executable

    # ── Virtual environment ───────────────────────────────────────────────
    report.in_venv, report.venv_type, report.venv_path = _detect_venv()

    # ── Package managers ──────────────────────────────────────────────────
    report.uv_available   = bool(shutil.which("uv"))
    report.pip_available  = bool(shutil.which("pip"))
    report.npm_available  = bool(shutil.which("npm"))
    report.pnpm_available = bool(shutil.which("pnpm"))
    report.yarn_available = bool(shutil.which("yarn"))
    report.cargo_available = bool(shutil.which("cargo"))
    report.go_available   = bool(shutil.which("go"))

    # ── Language runtime versions ─────────────────────────────────────────
    report.node_version = _get_version("node", "--version")
    report.rust_version = _get_version("rustc", "--version")
    report.go_version   = _get_version("go", "version")

    # ── Docker ────────────────────────────────────────────────────────────
    report.docker_running = _is_docker_running()

    # ── System service tools (postgres, redis, …) ─────────────────────────
    report.system_tools = check_system_tools(
        docker_running=report.docker_running,
        os_name=report.os_name,
    )

    # ── Summary log ───────────────────────────────────────────────────────
    tool_found   = sum(1 for t in report.system_tools if t.found)
    tool_missing = len(report.system_tools) - tool_found
    console.print(
        f"[green]  ✓ Environment: {report.os_name} / Python {report.python_version}, "
        f"venv={report.venv_type or 'none'}, "
        f"docker={'✓' if report.docker_running else '✗'}, "
        f"system tools {tool_found} found / {tool_missing} missing[/green]"
    )

    return report


# ===========================================================================
# Internal Detection Helpers
# ===========================================================================

def _detect_venv() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Detect whether a virtual environment is active and which kind.

    Priority order:
      1. Conda  — CONDA_DEFAULT_ENV is set and not 'base'
      2. venv   — sys.prefix differs from sys.base_prefix
      3. VIRTUAL_ENV env var — some tools set this without touching sys.prefix

    Returns:
        (in_venv, venv_type, venv_path)
        venv_type is 'conda', 'venv', or None.
        venv_path is the environment's root directory or None.
    """
    # ── Conda ──────────────────────────────────────────────────────────────
    conda_env    = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_env and conda_env != "base":
        return True, "conda", conda_prefix or None

    # ── Standard venv / virtualenv ─────────────────────────────────────────
    try:
        if sys.prefix != sys.base_prefix:
            return True, "venv", sys.prefix
    except AttributeError:
        # sys.base_prefix may not exist in some embedded Python builds
        pass

    # ── VIRTUAL_ENV env var fallback ───────────────────────────────────────
    venv_path = os.environ.get("VIRTUAL_ENV", "")
    if venv_path:
        return True, "venv", venv_path

    return False, None, None


def _get_version(binary: str, *args: str) -> Optional[str]:
    """
    Run ``binary *args`` and return the first non-empty output line.

    Args:
        binary: Command to run (e.g. 'node', 'rustc', 'go').
        *args:  Arguments to pass (e.g. '--version', 'version').

    Returns:
        Stripped first output line, or None if the binary is absent,
        the command fails, or it times out.
    """
    if not shutil.which(binary):
        return None
    try:
        result = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Some tools print to stderr on success (e.g. rustc --version)
        output = result.stdout or result.stderr
        for line in output.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        pass
    return None


def _is_docker_running() -> bool:
    """
    Return True if the Docker daemon is reachable.

    Runs ``docker info`` which exits 0 only when the daemon is running.
    A 5-second timeout prevents stalling on misconfigured Docker installs.
    """
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return False
