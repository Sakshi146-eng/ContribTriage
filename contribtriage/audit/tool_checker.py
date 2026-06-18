"""
contribtriage/audit/tool_checker.py

Stage 3 — System Service Tool Detection.

Checks whether common infrastructure services (PostgreSQL, Redis, MongoDB,
MySQL) are available on the contributor's PATH.

For each tool:
  - If found on PATH → report its path and version
  - If missing but Docker is running → surface a ready-to-run `docker run` snippet
  - Always → include an OS-specific package manager install command

Designed to be extended: add new entries to _TOOL_CONFIG to cover additional
services (e.g. Elasticsearch, RabbitMQ, Kafka) with zero code changes.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Dict, List, Optional

from contribtriage.models import SystemToolStatus


# ===========================================================================
# Tool Configuration Table
# ===========================================================================

_TOOL_CONFIG: Dict[str, dict] = {

    "postgres": {
        # Binaries to probe — first found wins
        "binaries":     ["psql", "postgres"],
        # Command to get a human-readable version string
        "version_cmd":  ["psql", "--version"],
        # Docker one-liner if daemon is running
        "docker_run":   (
            "docker run -d --name postgres "
            "-p 5432:5432 "
            "-e POSTGRES_PASSWORD=secret "
            "postgres:15"
        ),
        # OS-specific install hints
        "install": {
            "Windows": "winget install PostgreSQL.PostgreSQL",
            "Linux":   "sudo apt install postgresql",
            "Darwin":  "brew install postgresql",
        },
    },

    "redis": {
        "binaries":     ["redis-cli", "redis-server"],
        "version_cmd":  ["redis-cli", "--version"],
        "docker_run":   (
            "docker run -d --name redis "
            "-p 6379:6379 "
            "redis:7"
        ),
        "install": {
            "Windows": "winget install Redis.Redis",
            "Linux":   "sudo apt install redis-server",
            "Darwin":  "brew install redis",
        },
    },

    "mongodb": {
        "binaries":     ["mongod", "mongosh", "mongo"],
        "version_cmd":  ["mongod", "--version"],
        "docker_run":   (
            "docker run -d --name mongo "
            "-p 27017:27017 "
            "mongo:7"
        ),
        "install": {
            "Windows": "winget install MongoDB.Server",
            "Linux":   "sudo apt install mongodb",
            "Darwin":  "brew install mongodb-community",
        },
    },

    "mysql": {
        "binaries":     ["mysql", "mysqld"],
        "version_cmd":  ["mysql", "--version"],
        "docker_run":   (
            "docker run -d --name mysql "
            "-p 3306:3306 "
            "-e MYSQL_ROOT_PASSWORD=secret "
            "mysql:8"
        ),
        "install": {
            "Windows": "winget install Oracle.MySQL",
            "Linux":   "sudo apt install mysql-server",
            "Darwin":  "brew install mysql",
        },
    },

    "elasticsearch": {
        "binaries":     ["elasticsearch"],
        "version_cmd":  ["elasticsearch", "--version"],
        "docker_run":   (
            "docker run -d --name elasticsearch "
            "-p 9200:9200 "
            "-e discovery.type=single-node "
            "elasticsearch:8.12.0"
        ),
        "install": {
            "Windows": "https://www.elastic.co/downloads/elasticsearch",
            "Linux":   "sudo apt install elasticsearch",
            "Darwin":  "brew install elastic/tap/elasticsearch-full",
        },
    },

    "rabbitmq": {
        "binaries":     ["rabbitmq-server", "rabbitmqctl"],
        "version_cmd":  ["rabbitmqctl", "version"],
        "docker_run":   (
            "docker run -d --name rabbitmq "
            "-p 5672:5672 -p 15672:15672 "
            "rabbitmq:3-management"
        ),
        "install": {
            "Windows": "winget install RabbitMQ.RabbitMQ",
            "Linux":   "sudo apt install rabbitmq-server",
            "Darwin":  "brew install rabbitmq",
        },
    },
}


# ===========================================================================
# Public API
# ===========================================================================

def check_system_tools(
    docker_running: bool,
    os_name: str = "",
    tools: Optional[List[str]] = None,
) -> List[SystemToolStatus]:
    """
    Check the availability of common infrastructure services.

    Args:
        docker_running: Whether the Docker daemon is currently reachable.
                        Used to decide whether to surface docker run snippets.
        os_name:        Platform string ('Windows', 'Linux', 'Darwin').
                        Used to select the correct install command.
        tools:          Explicit list of tool names to check (keys in
                        _TOOL_CONFIG). Defaults to all configured tools.

    Returns:
        List of SystemToolStatus — one per checked tool, in insertion order.
    """
    tools_to_check = tools or list(_TOOL_CONFIG.keys())
    results: List[SystemToolStatus] = []

    for name in tools_to_check:
        cfg = _TOOL_CONFIG.get(name)
        if not cfg:
            continue
        results.append(_check_one_tool(name, cfg, docker_running, os_name))

    return results


# ===========================================================================
# Internal Helpers
# ===========================================================================

def _check_one_tool(
    name: str,
    cfg: dict,
    docker_running: bool,
    os_name: str,
) -> SystemToolStatus:
    """Probe a single tool and build its SystemToolStatus."""

    # ── Binary detection ──────────────────────────────────────────────────
    found_path: Optional[str] = None
    for binary in cfg.get("binaries", []):
        p = shutil.which(binary)
        if p:
            found_path = p
            break

    found = found_path is not None

    # ── Version string ────────────────────────────────────────────────────
    version: Optional[str] = None
    if found:
        version = _run_version_cmd(cfg.get("version_cmd", []))

    # ── Docker snippet (only when tool is missing and Docker is running) ──
    docker_snippet: Optional[str] = None
    if not found and docker_running:
        docker_snippet = cfg.get("docker_run")

    # ── OS-specific install command ───────────────────────────────────────
    os_install_cmd: Optional[str] = cfg.get("install", {}).get(os_name)

    # ── User-facing action string ─────────────────────────────────────────
    user_action: Optional[str] = None
    if not found:
        if docker_snippet:
            user_action = f"Docker available — run: {docker_snippet}"
        elif os_install_cmd:
            user_action = f"Install with: {os_install_cmd}"
        else:
            user_action = (
                f"Install {name} manually and ensure it is on PATH. "
                f"See: https://docs.{name}.org"
            )

    return SystemToolStatus(
        name=name,
        found=found,
        path=found_path,
        version=version,
        docker_snippet=docker_snippet,
        os_install_cmd=os_install_cmd,
        user_action=user_action,
    )


def _run_version_cmd(cmd: List[str]) -> Optional[str]:
    """
    Run a version-detection command and return the first output line.

    Returns None on any failure (missing binary, timeout, non-zero exit).
    """
    if not cmd:
        return None
    if not shutil.which(cmd[0]):
        return None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout or result.stderr
        for line in output.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        pass
    return None
