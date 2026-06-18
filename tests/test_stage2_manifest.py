"""
tests/test_stage2_manifest.py

Stage 2 test suite: Multi-ecosystem Manifest & Dependency Parser.

Coverage:
  - requirements.txt: dep extraction, comment skipping, edge cases
  - pyproject.toml: [project.dependencies], optional-deps, pytest detection,
                    python_version_req, regex fallback
  - package.json: deps, devDeps, node version, jest detection, npm test fallback
  - Cargo.toml: all three dep sections (dependencies / dev / build)
  - go.mod: block require, indirect markers
  - CONTRIBUTING.md: handled by Stage 1 Qdrant (not Stage 2)
  - Docker presence detection
  - Test directory discovery
  - parse_manifests integration: Python-only, Node-only, Rust-only, Go-only,
    hybrid polyglot repo
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contribtriage.ingestion.doc_reader import (
    _add_dep,
    _add_ecosystem,
    _add_test_framework,
    _find_test_dirs,
    _parse_cargo_toml,
    _parse_go_mod,
    _parse_package_json,
    _parse_pyproject,
    _parse_requirements_txt,
    parse_manifests,
)
from contribtriage.models import ProjectMeta

MANIFESTS = Path(__file__).parent / "fixtures" / "manifests"


def _fresh_meta() -> ProjectMeta:
    """Return a blank ProjectMeta for isolation between tests."""
    return ProjectMeta(repo_root=".")


# ===========================================================================
# 1. requirements.txt Parser
# ===========================================================================

class TestRequirementsTxt:

    def setup_method(self):
        self.meta = _fresh_meta()
        _parse_requirements_txt(MANIFESTS / "requirements.txt", self.meta)

    def test_extracts_runtime_deps(self):
        assert "requests" in self.meta.declared_deps
        assert "networkx" in self.meta.declared_deps
        assert "rich" in self.meta.declared_deps

    def test_extracts_dev_deps(self):
        assert "pytest" in self.meta.declared_deps
        assert "pytest-cov" in self.meta.declared_deps

    def test_strips_version_specifiers(self):
        # Versions like >=2.28.0 must be stripped; only the name survives
        assert "requests>=2.28.0" not in self.meta.declared_deps
        assert "requests" in self.meta.declared_deps

    def test_skips_comments(self):
        # Lines starting with # must not appear as deps
        for dep in self.meta.declared_deps:
            assert not dep.startswith("#")

    def test_skips_blank_lines(self):
        assert "" not in self.meta.declared_deps

    def test_ecosystem_is_python(self):
        for dep in ["requests", "networkx"]:
            assert self.meta.dep_ecosystem.get(dep) == "python"

    def test_no_duplicates(self):
        assert len(self.meta.declared_deps) == len(set(self.meta.declared_deps))

    def test_missing_file_does_not_raise(self, tmp_path):
        meta = _fresh_meta()
        # Should log a warning but never raise
        _parse_requirements_txt(tmp_path / "nonexistent.txt", meta)
        assert meta.declared_deps == []


# ===========================================================================
# 2. pyproject.toml Parser
# ===========================================================================

class TestPyprojectToml:

    def setup_method(self):
        self.meta = _fresh_meta()
        _parse_pyproject(MANIFESTS / "pyproject.toml", self.meta)

    def test_extracts_project_deps(self):
        assert "fastapi" in self.meta.declared_deps
        assert "uvicorn" in self.meta.declared_deps
        assert "sqlalchemy" in self.meta.declared_deps

    def test_extracts_optional_deps(self):
        # [project.optional-dependencies.dev] section
        assert "pytest" in self.meta.declared_deps
        assert "httpx" in self.meta.declared_deps

    def test_strips_version_from_dep_string(self):
        assert "fastapi>=0.100.0" not in self.meta.declared_deps
        assert "fastapi" in self.meta.declared_deps

    def test_detects_python_version_req(self):
        assert self.meta.python_version_req == ">=3.10"

    def test_detects_pytest_from_tool_section(self):
        # [tool.pytest.ini_options] present → pytest added to test_framework
        assert "pytest" in self.meta.test_framework

    def test_ecosystem_is_python(self):
        assert self.meta.dep_ecosystem.get("fastapi") == "python"

    def test_no_duplicates(self):
        assert len(self.meta.declared_deps) == len(set(self.meta.declared_deps))

    def test_missing_file_does_not_raise(self, tmp_path):
        meta = _fresh_meta()
        _parse_pyproject(tmp_path / "nonexistent.toml", meta)
        assert meta.declared_deps == []

    def test_malformed_toml_falls_back_gracefully(self, tmp_path):
        """A file with broken TOML should not raise — partial results are OK."""
        bad = tmp_path / "pyproject.toml"
        bad.write_text('[project\ndependencies = ["requests"', encoding="utf-8")
        meta = _fresh_meta()
        _parse_pyproject(bad, meta)  # must not raise


# ===========================================================================
# 3. package.json Parser
# ===========================================================================

class TestPackageJson:

    def setup_method(self):
        self.meta = _fresh_meta()
        _parse_package_json(MANIFESTS / "package.json", self.meta)

    def test_extracts_runtime_deps(self):
        assert "axios" in self.meta.declared_deps
        assert "express" in self.meta.declared_deps
        assert "lodash" in self.meta.declared_deps

    def test_extracts_dev_deps(self):
        assert "jest" in self.meta.declared_deps
        assert "typescript" in self.meta.declared_deps
        assert "eslint" in self.meta.declared_deps

    def test_detects_node_version(self):
        assert self.meta.node_version_req == ">=18.0.0"

    def test_detects_jest_test_runner(self):
        assert "npm test" in self.meta.test_framework

    def test_ecosystem_is_node(self):
        assert self.meta.dep_ecosystem.get("axios") == "node"
        assert self.meta.dep_ecosystem.get("jest") == "node"

    def test_no_duplicates(self):
        assert len(self.meta.declared_deps) == len(set(self.meta.declared_deps))

    def test_npm_test_fallback(self, tmp_path):
        """If no known framework in deps but test script exists → npm test."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            '{"scripts": {"test": "mocha"}, "dependencies": {}, "devDependencies": {}}',
            encoding="utf-8",
        )
        meta = _fresh_meta()
        _parse_package_json(pkg, meta)
        assert "npm test" in meta.test_framework

    def test_missing_file_does_not_raise(self, tmp_path):
        meta = _fresh_meta()
        _parse_package_json(tmp_path / "nonexistent.json", meta)
        assert meta.declared_deps == []

    def test_malformed_json_does_not_raise(self, tmp_path):
        bad = tmp_path / "package.json"
        bad.write_text("{invalid json", encoding="utf-8")
        meta = _fresh_meta()
        _parse_package_json(bad, meta)  # must not raise


# ===========================================================================
# 4. Cargo.toml Parser
# ===========================================================================

class TestCargoToml:

    def setup_method(self):
        self.meta = _fresh_meta()
        _parse_cargo_toml(MANIFESTS / "Cargo.toml", self.meta)

    def test_extracts_runtime_deps(self):
        assert "serde" in self.meta.declared_deps
        assert "serde_json" in self.meta.declared_deps
        assert "tokio" in self.meta.declared_deps
        assert "reqwest" in self.meta.declared_deps

    def test_extracts_dev_deps(self):
        assert "mockito" in self.meta.declared_deps
        assert "tempfile" in self.meta.declared_deps

    def test_extracts_build_deps(self):
        assert "cc" in self.meta.declared_deps

    def test_ecosystem_is_rust(self):
        assert self.meta.dep_ecosystem.get("serde") == "rust"
        assert self.meta.dep_ecosystem.get("mockito") == "rust"

    def test_no_duplicates(self):
        assert len(self.meta.declared_deps) == len(set(self.meta.declared_deps))

    def test_missing_file_does_not_raise(self, tmp_path):
        meta = _fresh_meta()
        _parse_cargo_toml(tmp_path / "nonexistent.toml", meta)
        assert meta.declared_deps == []


# ===========================================================================
# 5. go.mod Parser
# ===========================================================================

class TestGoMod:

    def setup_method(self):
        self.meta = _fresh_meta()
        _parse_go_mod(MANIFESTS / "go.mod", self.meta)

    def test_extracts_block_deps(self):
        # From the require ( ... ) block
        assert "github.com/gin-gonic/gin" in self.meta.declared_deps
        assert "github.com/go-redis/redis/v9" in self.meta.declared_deps
        assert "go.uber.org/zap" in self.meta.declared_deps

    def test_extracts_indirect_deps(self):
        # Indirect deps still get listed
        assert "github.com/stretchr/testify" in self.meta.declared_deps

    def test_ecosystem_is_go(self):
        assert self.meta.dep_ecosystem.get("github.com/gin-gonic/gin") == "go"

    def test_no_duplicates(self):
        assert len(self.meta.declared_deps) == len(set(self.meta.declared_deps))

    def test_missing_file_does_not_raise(self, tmp_path):
        meta = _fresh_meta()
        _parse_go_mod(tmp_path / "nonexistent.mod", meta)
        assert meta.declared_deps == []

    def test_single_line_require(self, tmp_path):
        """Single-line `require pkg v1.0.0` outside a block should be parsed."""
        mod = tmp_path / "go.mod"
        mod.write_text(
            "module example.com/app\ngo 1.21\nrequire github.com/some/pkg v1.2.3\n",
            encoding="utf-8",
        )
        meta = _fresh_meta()
        _parse_go_mod(mod, meta)
        assert "github.com/some/pkg" in meta.declared_deps


# ===========================================================================
# 6. Docker Detection
# ===========================================================================

class TestDockerDetection:

    def test_docker_detected_from_dockerfile(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.11", encoding="utf-8")
        meta = parse_manifests(str(tmp_path))
        assert meta.has_docker is True

    def test_docker_detected_from_compose(self, tmp_path):
        (tmp_path / "docker-compose.yml").write_text(
            "version: '3'\nservices:\n  app:\n    image: python:3.11",
            encoding="utf-8",
        )
        meta = parse_manifests(str(tmp_path))
        assert meta.has_docker is True

    def test_no_docker_when_absent(self, tmp_path):
        meta = parse_manifests(str(tmp_path))
        assert meta.has_docker is False

    def test_contributing_md_does_not_populate_any_field(self, tmp_path):
        """CONTRIBUTING.md is Stage 1's responsibility (Qdrant). Stage 2 must
        not read it or store it anywhere in ProjectMeta."""
        (tmp_path / "CONTRIBUTING.md").write_text(
            "# Contributing\n\nPlease read this.", encoding="utf-8"
        )
        meta = parse_manifests(str(tmp_path))
        # ProjectMeta has no contrib_guidelines field anymore
        assert not hasattr(meta, "contrib_guidelines")


# ===========================================================================
# 7. Test Directory Discovery
# ===========================================================================

class TestFindTestDirs:

    def test_finds_tests_dir(self, tmp_path):
        (tmp_path / "tests").mkdir()
        dirs = _find_test_dirs(tmp_path)
        assert any("tests" in d for d in dirs)

    def test_finds_test_dir(self, tmp_path):
        (tmp_path / "test").mkdir()
        dirs = _find_test_dirs(tmp_path)
        assert any("test" in d for d in dirs)

    def test_finds_jest_tests_dir(self, tmp_path):
        (tmp_path / "__tests__").mkdir()
        dirs = _find_test_dirs(tmp_path)
        assert any("__tests__" in d for d in dirs)

    def test_returns_empty_when_no_test_dirs(self, tmp_path):
        dirs = _find_test_dirs(tmp_path)
        assert dirs == []

    def test_multiple_test_dirs_found(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "spec").mkdir()
        dirs = _find_test_dirs(tmp_path)
        assert len(dirs) >= 2


# ===========================================================================
# 8. parse_manifests Integration Tests
# ===========================================================================

class TestParseManifestsIntegration:

    def test_python_only_repo(self, tmp_path):
        """Repo with only requirements.txt → python ecosystem detected."""
        req = tmp_path / "requirements.txt"
        req.write_text("flask>=2.0\ngunicorn>=20.0\n", encoding="utf-8")
        meta = parse_manifests(str(tmp_path))
        assert "python" in meta.ecosystems
        assert "flask" in meta.declared_deps
        assert "gunicorn" in meta.declared_deps

    def test_node_only_repo(self, tmp_path):
        """Repo with only package.json → node ecosystem detected."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            '{"dependencies": {"react": "^18.0.0"}, "devDependencies": {"jest": "^29.0.0"}}',
            encoding="utf-8",
        )
        meta = parse_manifests(str(tmp_path))
        assert "node" in meta.ecosystems
        assert "react" in meta.declared_deps
        assert "npm test" in meta.test_framework

    def test_rust_only_repo(self, tmp_path):
        """Repo with only Cargo.toml → rust ecosystem + cargo test."""
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text(
            '[package]\nname="myapp"\nversion="0.1.0"\nedition="2021"\n'
            '[dependencies]\nserde = "1.0"\n',
            encoding="utf-8",
        )
        meta = parse_manifests(str(tmp_path))
        assert "rust" in meta.ecosystems
        assert "serde" in meta.declared_deps
        assert "cargo test" in meta.test_framework

    def test_go_only_repo(self, tmp_path):
        """Repo with only go.mod → go ecosystem + go test."""
        go_mod = tmp_path / "go.mod"
        go_mod.write_text(
            "module example.com/app\ngo 1.21\nrequire (\n\tgithub.com/gin-gonic/gin v1.9.1\n)\n",
            encoding="utf-8",
        )
        meta = parse_manifests(str(tmp_path))
        assert "go" in meta.ecosystems
        assert "github.com/gin-gonic/gin" in meta.declared_deps
        assert "go test" in meta.test_framework

    def test_polyglot_repo_detects_all_ecosystems(self, tmp_path):
        """Hybrid repo (Python + Node + Rust) should detect all three."""
        (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
        (tmp_path / "package.json").write_text(
            '{"dependencies": {"axios": "^1.0"}, "devDependencies": {}}',
            encoding="utf-8",
        )
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname="app"\nversion="0.1.0"\nedition="2021"\n'
            '[dependencies]\ntokio = "1.0"\n',
            encoding="utf-8",
        )
        meta = parse_manifests(str(tmp_path))
        assert "python" in meta.ecosystems
        assert "node" in meta.ecosystems
        assert "rust" in meta.ecosystems
        # Deps from all ecosystems present
        assert "flask" in meta.declared_deps
        assert "axios" in meta.declared_deps
        assert "tokio" in meta.declared_deps

    def test_empty_repo_returns_empty_meta(self, tmp_path):
        """A repo with no manifests should return a valid but empty ProjectMeta."""
        meta = parse_manifests(str(tmp_path))
        assert meta.declared_deps == []
        assert meta.ecosystems == []
        assert meta.test_framework == []
        assert meta.has_docker is False

    def test_repo_root_set_correctly(self, tmp_path):
        meta = parse_manifests(str(tmp_path))
        assert meta.repo_root == str(tmp_path.resolve())

    def test_manifests_fixture_dir_full_parse(self):
        """Parse the full manifests fixtures dir — should find all 4 ecosystems."""
        meta = parse_manifests(str(MANIFESTS))
        # All four manifests are present
        assert "python" in meta.ecosystems
        assert "node" in meta.ecosystems
        assert "rust" in meta.ecosystems
        assert "go" in meta.ecosystems
        # Sanity-check a dep from each
        assert "fastapi" in meta.declared_deps   # from pyproject.toml
        assert "axios" in meta.declared_deps     # from package.json
        assert "serde" in meta.declared_deps     # from Cargo.toml
        assert "github.com/gin-gonic/gin" in meta.declared_deps  # from go.mod

    def test_no_duplicate_deps_in_polyglot(self, tmp_path):
        """Running parse_manifests on a polyglot repo produces no duplicate dep names."""
        (tmp_path / "requirements.txt").write_text("requests\nflask\n", encoding="utf-8")
        (tmp_path / "package.json").write_text(
            '{"dependencies": {"axios": "^1.0"}, "devDependencies": {}}',
            encoding="utf-8",
        )
        meta = parse_manifests(str(tmp_path))
        assert len(meta.declared_deps) == len(set(meta.declared_deps))
