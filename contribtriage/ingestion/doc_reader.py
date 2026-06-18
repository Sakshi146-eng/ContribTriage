"""
contribtriage/ingestion/doc_reader.py

Stage 2 — Manifest & Dependency Parsing.

Scans the repository root for all known manifest and config files across
Python, Node.js, Rust, and Go ecosystems. Extracts:

  - Declared runtime + dev dependencies (names only, not version pins)
  - Which ecosystem each dependency belongs to
  - Detected test frameworks and runners
  - Test directory paths
  - Contributing guidelines text
  - Python / Node version requirements
  - Docker presence

Returns a populated ProjectMeta dataclass that feeds the LangGraph state.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from contribtriage.models import ProjectMeta

console = Console()


# ===========================================================================
# Manifest file names
# ===========================================================================

_PYTHON_REQ_FILES = {
    "requirements.txt", "requirements-dev.txt", "requirements_dev.txt",
    "requirements-test.txt", "requirements_test.txt",
}
_PYPROJECT_FILE   = "pyproject.toml"
_NODE_FILE        = "package.json"
_RUST_FILE        = "Cargo.toml"
_GO_FILE          = "go.mod"
_DOCKER_FILES     = {"Dockerfile", "docker-compose.yml", "docker-compose.yaml"}

# Test directory names to look for
_TEST_DIR_NAMES   = {"tests", "test", "__tests__", "spec", "specs", "e2e"}

# Node test framework → runner command
_NODE_TEST_RUNNERS = {
    "jest":     "npm test",
    "mocha":    "npm test",
    "vitest":   "npm test",
    "jasmine":  "npm test",
    "@jest/core": "npm test",
}


# ===========================================================================
# Public API
# ===========================================================================

def parse_manifests(repo_root: str) -> ProjectMeta:
    """
    Scan *repo_root* for all manifest files and return a populated ProjectMeta.

    Parsing is best-effort — a missing or malformed manifest is logged and
    skipped; it never raises an exception.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        Populated ProjectMeta with declared deps, ecosystems, test frameworks,
        test dirs, contributing guidelines, and Docker presence.
    """
    root = Path(repo_root).resolve()
    meta = ProjectMeta(repo_root=str(root))

    # ── Python ────────────────────────────────────────────────────────────
    pyproject_path = root / _PYPROJECT_FILE
    if pyproject_path.exists():
        _parse_pyproject(pyproject_path, meta)
        _add_ecosystem(meta, "python")

    for req_name in _PYTHON_REQ_FILES:
        req_path = root / req_name
        if req_path.exists():
            _parse_requirements_txt(req_path, meta)
            _add_ecosystem(meta, "python")

    # Always add pytest if a Python ecosystem was detected and no framework set
    if "python" in meta.ecosystems and not any(
        f in meta.test_framework for f in ["pytest", "unittest"]
    ):
        meta.test_framework.append("pytest")

    # ── Node.js ───────────────────────────────────────────────────────────
    pkg_json_path = root / _NODE_FILE
    if pkg_json_path.exists():
        _parse_package_json(pkg_json_path, meta)
        _add_ecosystem(meta, "node")

    # ── Rust ──────────────────────────────────────────────────────────────
    cargo_path = root / _RUST_FILE
    if cargo_path.exists():
        _parse_cargo_toml(cargo_path, meta)
        _add_ecosystem(meta, "rust")
        _add_test_framework(meta, "cargo test")

    # ── Go ────────────────────────────────────────────────────────────────
    go_mod_path = root / _GO_FILE
    if go_mod_path.exists():
        _parse_go_mod(go_mod_path, meta)
        _add_ecosystem(meta, "go")
        _add_test_framework(meta, "go test")

    # ── Docker ───────────────────────────────────────────────────────────
    meta.has_docker = any((root / f).exists() for f in _DOCKER_FILES)

    # Note: CONTRIBUTING.md is intentionally skipped here.
    # Stage 1 (VectorStore.ingest_repo) already chunks and embeds all .md docs.
    # The LLM retrieves relevant guidelines on-demand via VectorStore.query().

    # ── Test directories ─────────────────────────────────────────────────
    meta.test_dirs = _find_test_dirs(root)

    console.print(
        f"[green]  ✓ Manifests: {len(meta.declared_deps)} deps across "
        f"ecosystems {meta.ecosystems or ['(none detected)']}, "
        f"test runners: {meta.test_framework or ['(none detected)']}[/green]"
    )

    return meta


# ===========================================================================
# Python Parsers
# ===========================================================================

def _parse_requirements_txt(file: Path, meta: ProjectMeta) -> None:
    """
    Parse a requirements.txt-style file.

    Handles:
      - Plain package names:          requests
      - Versioned:                    requests>=2.28.0
      - Extras:                       requests[security]>=2.28.0
      - Environment markers:          requests; python_version >= "3.9"
      - Git / URL installs:           skipped (start with git+/http)
      - Comments and blank lines:     skipped
      - -r / -c / -i flags:           skipped
    """
    try:
        content = file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        console.print(f"[yellow]  ⚠ Could not read {file.name}: {exc}[/yellow]")
        return

    for raw in content.splitlines():
        line = raw.strip()
        if (
            not line
            or line.startswith("#")
            or line.startswith("-")
            or line.startswith("git+")
            or line.startswith("http")
        ):
            continue
        # Strip version specifier, extras bracket, and environment markers
        pkg = re.split(r"[>=<!;\[\s]", line)[0].strip()
        if pkg:
            _add_dep(meta, pkg, "python")


def _parse_pyproject(file: Path, meta: ProjectMeta) -> None:
    """
    Parse pyproject.toml for dependencies, version requirements, and tooling.

    Tries tomllib (stdlib in 3.11+) then tomli (backport), then falls back
    to a regex scan so the function always produces partial results even if
    the TOML parser isn't available.
    """
    try:
        content = file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        console.print(f"[yellow]  ⚠ Could not read {file.name}: {exc}[/yellow]")
        return

    data = _parse_toml(content)

    if data is None:
        # Fallback: regex extraction of dep lines
        _regex_extract_deps(content, meta, "python")
        return

    project = data.get("project", {})

    # Runtime dependencies
    for dep_str in project.get("dependencies", []):
        pkg = re.split(r"[>=<!;\[\s]", dep_str)[0].strip()
        if pkg:
            _add_dep(meta, pkg, "python")

    # Optional / dev dependencies
    for group_deps in project.get("optional-dependencies", {}).values():
        for dep_str in group_deps:
            pkg = re.split(r"[>=<!;\[\s]", dep_str)[0].strip()
            if pkg:
                _add_dep(meta, pkg, "python")

    # [tool.poetry.dependencies] style (Poetry projects)
    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    for pkg in poetry_deps:
        if pkg.lower() != "python":
            _add_dep(meta, pkg, "python")

    # Python version requirement
    if "requires-python" in project:
        meta.python_version_req = project["requires-python"]

    # Detect pytest from [tool.pytest] or dev deps containing pytest
    tool_section = data.get("tool", {})
    if "pytest" in tool_section or "pytest.ini_options" in tool_section:
        _add_test_framework(meta, "pytest")

    if "pytest" in meta.declared_deps:
        _add_test_framework(meta, "pytest")


# ===========================================================================
# Node.js Parser
# ===========================================================================

def _parse_package_json(file: Path, meta: ProjectMeta) -> None:
    """Parse package.json for deps, dev deps, Node version, and test runner."""
    try:
        raw = file.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[yellow]  ⚠ Could not parse {file.name}: {exc}[/yellow]")
        return

    for pkg in data.get("dependencies", {}):
        _add_dep(meta, pkg, "node")

    for pkg in data.get("devDependencies", {}):
        _add_dep(meta, pkg, "node")

    # Node version requirement from engines field
    engines = data.get("engines", {})
    if "node" in engines:
        meta.node_version_req = engines["node"]

    # Detect test runner from devDependencies
    all_deps = {
        **data.get("dependencies", {}),
        **data.get("devDependencies", {}),
    }
    for framework, runner in _NODE_TEST_RUNNERS.items():
        if framework in all_deps:
            _add_test_framework(meta, runner)
            break
    else:
        # Check for a test script as a fallback
        if "test" in data.get("scripts", {}):
            _add_test_framework(meta, "npm test")

    # Detect pnpm / yarn from lockfiles (package.json dir)
    parent = file.parent
    if (parent / "pnpm-lock.yaml").exists():
        meta.extra_install_hints = getattr(meta, "extra_install_hints", [])
        meta.extra_install_hints.append("pnpm install")  # type: ignore[attr-defined]
    elif (parent / "yarn.lock").exists():
        meta.extra_install_hints = getattr(meta, "extra_install_hints", [])
        meta.extra_install_hints.append("yarn install")  # type: ignore[attr-defined]


# ===========================================================================
# Rust Parser
# ===========================================================================

def _parse_cargo_toml(file: Path, meta: ProjectMeta) -> None:
    """Parse Cargo.toml for crate dependencies."""
    try:
        content = file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        console.print(f"[yellow]  ⚠ Could not read {file.name}: {exc}[/yellow]")
        return

    data = _parse_toml(content)

    if data is None:
        _regex_extract_deps(content, meta, "rust")
        return

    # Standard dependency sections
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        for pkg in data.get(section, {}):
            _add_dep(meta, pkg, "rust")

    # Workspace dependencies (monorepos)
    workspace = data.get("workspace", {})
    for pkg in workspace.get("dependencies", {}):
        _add_dep(meta, pkg, "rust")


# ===========================================================================
# Go Parser
# ===========================================================================

def _parse_go_mod(file: Path, meta: ProjectMeta) -> None:
    """
    Parse go.mod for module dependencies.

    go.mod format:
      module github.com/user/repo
      go 1.21
      require (
          github.com/some/dep v1.2.3
          github.com/other/dep v0.4.0 // indirect
      )
      require github.com/single/dep v1.0.0
    """
    try:
        content = file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        console.print(f"[yellow]  ⚠ Could not read {file.name}: {exc}[/yellow]")
        return

    # Single-line require: `require github.com/pkg v1.2.3`
    for match in re.finditer(
        r"^require\s+([\w./\-]+)\s+v[\w.\-]+", content, re.MULTILINE
    ):
        _add_dep(meta, match.group(1), "go")

    # Block require: `require (\n\tpkg v...\n)`
    for block in re.finditer(r"require\s*\((.*?)\)", content, re.DOTALL):
        for line in block.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if parts:
                # Strip `// indirect` comments
                pkg = parts[0]
                if pkg:
                    _add_dep(meta, pkg, "go")


# ===========================================================================
# Shared Helpers
# ===========================================================================

def _parse_toml(content: str) -> Optional[dict]:
    """
    Parse TOML content string using tomllib (3.11+) or tomli backport.
    Returns None if neither is available or content is malformed.
    """
    try:
        if sys.version_info >= (3, 11):
            import tomllib
            return tomllib.loads(content)
        else:
            import tomli  # type: ignore[import]
            return tomli.loads(content)
    except Exception:
        return None


def _regex_extract_deps(content: str, meta: ProjectMeta, ecosystem: str) -> None:
    """
    Regex fallback: extract quoted strings from a TOML dependency table.
    Catches simple cases like:  requests = "^2.28"  or  "requests>=2.0"
    """
    for match in re.finditer(r'"([A-Za-z][A-Za-z0-9_\-\.]+)', content):
        candidate = match.group(1)
        # Exclude version strings (start with digit) and long paths
        if not candidate[0].isdigit() and "/" not in candidate and len(candidate) < 50:
            _add_dep(meta, candidate, ecosystem)


def _add_dep(meta: ProjectMeta, pkg: str, ecosystem: str) -> None:
    """Add a dependency if not already present."""
    if pkg and pkg not in meta.declared_deps:
        meta.declared_deps.append(pkg)
        meta.dep_ecosystem[pkg] = ecosystem


def _add_ecosystem(meta: ProjectMeta, name: str) -> None:
    """Add ecosystem label if not already present."""
    if name not in meta.ecosystems:
        meta.ecosystems.append(name)


def _add_test_framework(meta: ProjectMeta, runner: str) -> None:
    """Add test runner if not already present."""
    if runner not in meta.test_framework:
        meta.test_framework.append(runner)


def _find_test_dirs(root: Path) -> List[str]:
    """Return absolute paths of test directories found directly under root."""
    return [
        str(root / name)
        for name in _TEST_DIR_NAMES
        if (root / name).is_dir()
    ]
