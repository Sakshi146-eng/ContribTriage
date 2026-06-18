"""
tests/test_stage3_audit.py

Stage 3 test suite: Environment Auditor + System Tool Checker.

Strategy: all subprocess calls, shutil.which calls, and platform/os/sys values
are mocked via unittest.mock so the tests are hermetic — they produce the same
result on every machine regardless of what is actually installed.

Coverage:
  - _detect_venv(): conda env, standard venv, VIRTUAL_ENV fallback, no venv
  - _get_version(): binary absent, success path, stderr fallback, timeout
  - _is_docker_running(): docker absent, daemon up, daemon down, timeout
  - check_system_tools(): tool found with version, tool missing + Docker,
                          tool missing + no Docker, custom tool list,
                          os_install_cmd selection per OS
  - _check_one_tool(): unit-level user_action text
  - audit_environment(): full integration mock — all fields populated correctly
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from contribtriage.audit.env_auditor import (
    _detect_venv,
    _get_version,
    _is_docker_running,
    audit_environment,
)
from contribtriage.audit.tool_checker import (
    _TOOL_CONFIG,
    _check_one_tool,
    _run_version_cmd,
    check_system_tools,
)
from contribtriage.models import EnvReport, SystemToolStatus


# ===========================================================================
# Helpers
# ===========================================================================

def _make_completed(stdout: str = "", returncode: int = 0) -> MagicMock:
    """Return a mock CompletedProcess-like object."""
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = returncode
    return m


# ===========================================================================
# 1. _detect_venv()
# ===========================================================================

class TestDetectVenv:

    def test_conda_env_detected(self):
        env = {"CONDA_DEFAULT_ENV": "myenv", "CONDA_PREFIX": "/opt/conda/envs/myenv"}
        with patch.dict(os.environ, env, clear=False):
            in_venv, venv_type, venv_path = _detect_venv()
        assert in_venv is True
        assert venv_type == "conda"
        assert venv_path == "/opt/conda/envs/myenv"

    def test_conda_base_env_not_counted(self):
        """CONDA_DEFAULT_ENV='base' should NOT be treated as an active venv."""
        env = {"CONDA_DEFAULT_ENV": "base", "CONDA_PREFIX": "/opt/conda"}
        with patch.dict(os.environ, env, clear=False):
            # Also ensure we're not in a real venv
            with patch.object(sys, "prefix", sys.base_prefix):
                in_venv, venv_type, _ = _detect_venv()
        # 'base' conda is ignored; check falls through to venv logic
        assert venv_type != "conda"

    def test_standard_venv_detected(self):
        """sys.prefix != sys.base_prefix → standard venv."""
        # Remove conda vars to avoid interference
        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith("CONDA")}
        clean_env.pop("VIRTUAL_ENV", None)
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.object(sys, "prefix", "/home/user/.venv"):
                with patch.object(sys, "base_prefix", "/usr"):
                    in_venv, venv_type, venv_path = _detect_venv()
        assert in_venv is True
        assert venv_type == "venv"
        assert venv_path == "/home/user/.venv"

    def test_virtual_env_var_fallback(self):
        """VIRTUAL_ENV env var alone triggers venv detection."""
        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith("CONDA")}
        clean_env["VIRTUAL_ENV"] = "/home/user/.venv"
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.object(sys, "prefix", sys.base_prefix):
                in_venv, venv_type, venv_path = _detect_venv()
        assert in_venv is True
        assert venv_type == "venv"
        assert venv_path == "/home/user/.venv"

    def test_no_venv_returns_false(self):
        """Clean environment with no venv markers → (False, None, None)."""
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("CONDA_DEFAULT_ENV", "CONDA_PREFIX", "VIRTUAL_ENV")}
        with patch.dict(os.environ, clean_env, clear=True):
            with patch.object(sys, "prefix", sys.base_prefix):
                in_venv, venv_type, venv_path = _detect_venv()
        assert in_venv is False
        assert venv_type is None
        assert venv_path is None


# ===========================================================================
# 2. _get_version()
# ===========================================================================

class TestGetVersion:

    def test_returns_none_when_binary_absent(self):
        with patch("contribtriage.audit.env_auditor.shutil.which", return_value=None):
            assert _get_version("node", "--version") is None

    def test_returns_stdout_first_line(self):
        with patch("contribtriage.audit.env_auditor.shutil.which", return_value="/usr/bin/node"):
            with patch("contribtriage.audit.env_auditor.subprocess.run",
                       return_value=_make_completed("v20.11.0\n", 0)):
                result = _get_version("node", "--version")
        assert result == "v20.11.0"

    def test_falls_back_to_stderr(self):
        """Some tools print version to stderr — must still be captured."""
        proc = _make_completed("", 0)
        proc.stderr = "rustc 1.78.0 (9b00956e5 2024-04-29)\n"
        with patch("contribtriage.audit.env_auditor.shutil.which", return_value="/usr/bin/rustc"):
            with patch("contribtriage.audit.env_auditor.subprocess.run", return_value=proc):
                result = _get_version("rustc", "--version")
        assert result == "rustc 1.78.0 (9b00956e5 2024-04-29)"

    def test_returns_none_on_timeout(self):
        with patch("contribtriage.audit.env_auditor.shutil.which", return_value="/usr/bin/node"):
            with patch("contribtriage.audit.env_auditor.subprocess.run",
                       side_effect=subprocess.TimeoutExpired("node", 5)):
                assert _get_version("node", "--version") is None

    def test_returns_none_on_os_error(self):
        with patch("contribtriage.audit.env_auditor.shutil.which", return_value="/usr/bin/go"):
            with patch("contribtriage.audit.env_auditor.subprocess.run",
                       side_effect=OSError("no such file")):
                assert _get_version("go", "version") is None

    def test_skips_empty_lines(self):
        """First non-empty line should be returned even if stdout starts with blank lines."""
        proc = _make_completed("\n\n  go version go1.22.1 linux/amd64\n", 0)
        with patch("contribtriage.audit.env_auditor.shutil.which", return_value="/usr/bin/go"):
            with patch("contribtriage.audit.env_auditor.subprocess.run", return_value=proc):
                result = _get_version("go", "version")
        assert result == "go version go1.22.1 linux/amd64"


# ===========================================================================
# 3. _is_docker_running()
# ===========================================================================

class TestIsDockerRunning:

    def test_returns_false_when_docker_absent(self):
        with patch("contribtriage.audit.env_auditor.shutil.which", return_value=None):
            assert _is_docker_running() is False

    def test_returns_true_when_daemon_up(self):
        with patch("contribtriage.audit.env_auditor.shutil.which", return_value="/usr/bin/docker"):
            with patch("contribtriage.audit.env_auditor.subprocess.run",
                       return_value=_make_completed("", 0)):
                assert _is_docker_running() is True

    def test_returns_false_when_daemon_down(self):
        with patch("contribtriage.audit.env_auditor.shutil.which", return_value="/usr/bin/docker"):
            with patch("contribtriage.audit.env_auditor.subprocess.run",
                       return_value=_make_completed("", 1)):
                assert _is_docker_running() is False

    def test_returns_false_on_timeout(self):
        with patch("contribtriage.audit.env_auditor.shutil.which", return_value="/usr/bin/docker"):
            with patch("contribtriage.audit.env_auditor.subprocess.run",
                       side_effect=subprocess.TimeoutExpired("docker", 5)):
                assert _is_docker_running() is False


# ===========================================================================
# 4. check_system_tools()
# ===========================================================================

class TestCheckSystemTools:

    def test_tool_found_returns_found_true(self):
        with patch("contribtriage.audit.tool_checker.shutil.which",
                   return_value="/usr/bin/psql"):
            with patch("contribtriage.audit.tool_checker.subprocess.run",
                       return_value=_make_completed("psql (PostgreSQL) 15.3", 0)):
                results = check_system_tools(
                    docker_running=False,
                    os_name="Linux",
                    tools=["postgres"],
                )
        assert len(results) == 1
        assert results[0].found is True
        assert results[0].name == "postgres"

    def test_version_extracted_when_found(self):
        with patch("contribtriage.audit.tool_checker.shutil.which",
                   return_value="/usr/bin/psql"):
            with patch("contribtriage.audit.tool_checker.subprocess.run",
                       return_value=_make_completed("psql (PostgreSQL) 15.3", 0)):
                results = check_system_tools(
                    docker_running=False,
                    os_name="Linux",
                    tools=["postgres"],
                )
        assert results[0].version == "psql (PostgreSQL) 15.3"

    def test_docker_snippet_when_missing_and_docker_running(self):
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            results = check_system_tools(
                docker_running=True,
                os_name="Linux",
                tools=["postgres"],
            )
        assert results[0].found is False
        assert results[0].docker_snippet is not None
        assert "docker run" in results[0].docker_snippet

    def test_no_docker_snippet_when_docker_not_running(self):
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            results = check_system_tools(
                docker_running=False,
                os_name="Linux",
                tools=["postgres"],
            )
        assert results[0].docker_snippet is None

    def test_linux_install_cmd_returned(self):
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            results = check_system_tools(
                docker_running=False,
                os_name="Linux",
                tools=["redis"],
            )
        assert results[0].os_install_cmd == "sudo apt install redis-server"

    def test_windows_install_cmd_returned(self):
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            results = check_system_tools(
                docker_running=False,
                os_name="Windows",
                tools=["redis"],
            )
        assert results[0].os_install_cmd == "winget install Redis.Redis"

    def test_macos_install_cmd_returned(self):
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            results = check_system_tools(
                docker_running=False,
                os_name="Darwin",
                tools=["mysql"],
            )
        assert results[0].os_install_cmd == "brew install mysql"

    def test_user_action_set_when_missing(self):
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            results = check_system_tools(
                docker_running=False,
                os_name="Linux",
                tools=["postgres"],
            )
        assert results[0].user_action is not None
        assert len(results[0].user_action) > 0

    def test_user_action_mentions_docker_when_available(self):
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            results = check_system_tools(
                docker_running=True,
                os_name="Linux",
                tools=["postgres"],
            )
        assert "Docker" in results[0].user_action or "docker" in results[0].user_action

    def test_no_user_action_when_tool_found(self):
        """When a tool is found, user_action should be None (no action needed)."""
        with patch("contribtriage.audit.tool_checker.shutil.which",
                   return_value="/usr/bin/psql"):
            with patch("contribtriage.audit.tool_checker.subprocess.run",
                       return_value=_make_completed("psql 15.3", 0)):
                results = check_system_tools(
                    docker_running=False,
                    os_name="Linux",
                    tools=["postgres"],
                )
        assert results[0].user_action is None

    def test_custom_tool_list_respected(self):
        """Passing tools=['redis'] should only return Redis status."""
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            results = check_system_tools(
                docker_running=False,
                os_name="Linux",
                tools=["redis"],
            )
        assert len(results) == 1
        assert results[0].name == "redis"

    def test_all_configured_tools_checked_by_default(self):
        """Default (tools=None) should check all tools in _TOOL_CONFIG."""
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            results = check_system_tools(docker_running=False, os_name="Linux")
        assert len(results) == len(_TOOL_CONFIG)

    def test_unknown_tool_name_skipped(self):
        """Passing a tool name not in _TOOL_CONFIG returns empty list."""
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            results = check_system_tools(
                docker_running=False,
                os_name="Linux",
                tools=["nonexistent_tool_xyz"],
            )
        assert results == []


# ===========================================================================
# 5. _run_version_cmd()
# ===========================================================================

class TestRunVersionCmd:

    def test_returns_none_for_empty_cmd(self):
        assert _run_version_cmd([]) is None

    def test_returns_none_when_binary_absent(self):
        with patch("contribtriage.audit.tool_checker.shutil.which", return_value=None):
            assert _run_version_cmd(["psql", "--version"]) is None

    def test_returns_first_line_of_output(self):
        with patch("contribtriage.audit.tool_checker.shutil.which",
                   return_value="/usr/bin/psql"):
            with patch("contribtriage.audit.tool_checker.subprocess.run",
                       return_value=_make_completed("psql (PostgreSQL) 15.3\n", 0)):
                result = _run_version_cmd(["psql", "--version"])
        assert result == "psql (PostgreSQL) 15.3"

    def test_returns_none_on_timeout(self):
        with patch("contribtriage.audit.tool_checker.shutil.which",
                   return_value="/usr/bin/psql"):
            with patch("contribtriage.audit.tool_checker.subprocess.run",
                       side_effect=subprocess.TimeoutExpired("psql", 5)):
                assert _run_version_cmd(["psql", "--version"]) is None


# ===========================================================================
# 6. audit_environment() Integration
# ===========================================================================

class TestAuditEnvironment:
    """
    Full integration test using patch to mock all I/O.
    Verifies that audit_environment() correctly maps helper outputs
    into EnvReport fields.
    """

    def _run_with_mocks(
        self,
        os_system: str = "Linux",
        os_version: str = "5.15.0",
        python_version: str = "3.11.4",
        python_path: str = "/usr/bin/python3",
        which_map: dict = None,
        subprocess_map: dict = None,
        node_ver: str = "v20.11.0",
        rust_ver: str = "rustc 1.78.0",
        go_ver: str = "go version go1.22.1",
        docker_up: bool = True,
    ) -> EnvReport:
        """Run audit_environment with a fully controlled environment."""
        # Use `is None` — NOT `or` — because `{}` (an intentionally empty map
        # passed by test_missing_package_manager_is_false) is falsy, and
        # `{} or default_dict` would silently discard it.
        if which_map is None:
            which_map = {
                "uv":    "/usr/bin/uv",
                "pip":   "/usr/bin/pip",
                "npm":   "/usr/bin/npm",
                "node":  "/usr/bin/node",
                "rustc": "/usr/bin/rustc",
                "go":    "/usr/bin/go",
                "cargo": "/usr/bin/cargo",
                "docker": "/usr/bin/docker",
            }

        def fake_which(name):
            return which_map.get(name)

        # Stub subprocess.run for version commands and docker info
        def fake_run(cmd, **kwargs):
            m = _make_completed("", 0)
            if "node" in cmd:
                m.stdout = node_ver
            elif "rustc" in cmd:
                m.stdout = rust_ver
            elif "go" in cmd and "version" in cmd:
                m.stdout = go_ver
            elif "docker" in cmd and "info" in cmd:
                m.returncode = 0 if docker_up else 1
            elif "psql" in cmd or "redis-cli" in cmd:
                m.returncode = 1  # simulate tools missing
            else:
                m.returncode = 1
            return m

        # Use global patches — both env_auditor and tool_checker import the
        # same shutil/subprocess singletons. Patching via two different dotted
        # paths simultaneously causes the second to silently overwrite the first
        # (both resolve to the same module object). One global patch covers both.
        with patch("shutil.which", side_effect=fake_which), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("contribtriage.audit.env_auditor.platform.system", return_value=os_system), \
             patch("contribtriage.audit.env_auditor.platform.version", return_value=os_version), \
             patch("contribtriage.audit.env_auditor.platform.python_version", return_value=python_version), \
             patch.object(sys, "executable", python_path):
            report = audit_environment("/tmp/test-repo")
        return report

    def test_os_fields_populated(self):
        report = self._run_with_mocks(os_system="Linux", os_version="5.15.0")
        assert report.os_name == "Linux"
        assert report.os_version == "5.15.0"

    def test_python_fields_populated(self):
        report = self._run_with_mocks(python_version="3.11.4")
        assert report.python_version == "3.11.4"
        assert report.python_path == "/usr/bin/python3"

    def test_package_managers_detected(self):
        report = self._run_with_mocks()
        assert report.uv_available is True
        assert report.pip_available is True
        assert report.npm_available is True
        assert report.cargo_available is True
        assert report.go_available is True

    def test_missing_package_manager_is_false(self):
        report = self._run_with_mocks(which_map={})
        assert report.uv_available is False
        assert report.pip_available is False
        assert report.npm_available is False

    def test_docker_running_true(self):
        report = self._run_with_mocks(docker_up=True)
        assert report.docker_running is True

    def test_docker_running_false(self):
        report = self._run_with_mocks(docker_up=False)
        assert report.docker_running is False

    def test_node_version_captured(self):
        report = self._run_with_mocks(node_ver="v20.11.0")
        assert report.node_version == "v20.11.0"

    def test_returns_env_report_type(self):
        report = self._run_with_mocks()
        assert isinstance(report, EnvReport)

    def test_system_tools_list_populated(self):
        report = self._run_with_mocks()
        assert isinstance(report.system_tools, list)
        # At least one tool is always checked
        assert len(report.system_tools) >= 1

    def test_never_raises_on_all_missing(self):
        """Even when nothing is installed, audit_environment must not raise."""
        with patch("shutil.which", return_value=None), \
             patch("subprocess.run", side_effect=OSError("not found")):
            report = audit_environment("/tmp/empty-repo")
        assert isinstance(report, EnvReport)
        assert report.python_version  # Python version always known (no subprocess needed)
