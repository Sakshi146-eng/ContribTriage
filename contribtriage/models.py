"""
contribtriage/models.py

Shared dataclasses + LangGraph TypedDict state schema.

Two layers of types live here:
  1. Dataclasses  — rich, IDE-friendly types used within each stage module.
  2. LangGraphState (TypedDict) — the single state object threaded through
     every LangGraph node. Dataclasses are nested inside it as values.

Separation rationale: LangGraph requires TypedDict (not dataclasses) for
its state schema so it can do partial-dict merging between nodes. We keep
the internal stage models as dataclasses for clarity and type safety.
"""

from __future__ import annotations

import enum
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    TypedDict,
)


# ===========================================================================
# Pipeline Stage Enum
# ===========================================================================

class PipelineStage(enum.Enum):
    """Ordered stages of the ContribTriage pipeline."""
    INGEST          = "ingest"
    MANIFEST        = "manifest"
    AUDIT           = "audit"
    LLM_BRAIN       = "llm_brain"
    CHECK_COVERAGE  = "check_coverage"
    GENERATE_TESTS  = "generate_tests"
    RUN_TESTS       = "run_tests"
    ANALYZE_FAILURE = "analyze_failure"
    SYSTEM_DEP      = "system_dep"
    APP_DEP         = "app_dep"
    SELECTIVE_RERUN = "selective_rerun"
    REPORT          = "report"
    DONE            = "done"


# ===========================================================================
# Stage 1 — Ingestion Models
# ===========================================================================

@dataclass
class ModuleNode:
    """
    Represents a single source file in the knowledge graph.
    Works for Python, JavaScript, TypeScript, Rust, and Go files.

    Attributes:
        path:        Absolute path to the file.
        module_name: Dot-separated logical name (e.g. 'contribtriage.cli').
        language:    Detected language ('Python', 'JavaScript', 'Rust', 'Go', etc.)
        classes:     Class / struct / trait / interface names defined in this file.
        functions:   Function / method names defined in this file.
        imports:     External modules this file imports / requires / uses.
        todos:       TODO/FIXME/BUG comment strings found in this file.
        docstring:   Module-level docstring or top-of-file comment (if present).
    """
    path: str
    module_name: str
    language: str = "Python"
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    todos: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class NonPythonFile:
    """
    Lightweight record for a non-code or non-primary file in the repository.
    These are ingested into the Qdrant vector store but not structurally graphed.

    Attributes:
        path:      Absolute path to the file.
        ext:       File extension (e.g. '.yml', '.md', '.sh', '').
        category:  Human-readable category:
                     'docs'     — README, CONTRIBUTING, .md, .rst
                     'ci'       — GitHub Actions, .yml/.yaml
                     'docker'   — Dockerfile, docker-compose
                     'config'   — .json, .toml, .ini, .cfg
                     'frontend' — .css, .html
                     'data'     — .csv, .json data files
                     'other'    — everything else
        size_kb:   File size in kilobytes (rounded).
    """
    path: str
    ext: str
    category: str = "other"
    size_kb: float = 0.0


@dataclass
class KnowledgeGraph:
    """
    Full structural map of the repository.

    The repo is treated as polyglot:
      - Python/JS/TS/Rust/Go files → AST/regex parsed → NetworkX nodes + edges
      - All other text files → tracked in non_python_files → Qdrant vector store

    Attributes:
        nodes:             Parsed ModuleNodes keyed by module_name.
        edges:             (source, target, relation) tuples.
                           relation ∈ {'imports', 'calls', 'inherits', 'contains'}
        uncovered_funcs:   'module.function' strings with zero test references.
        graph_json_path:   Path to serialized NetworkX graph (node-link JSON).
        non_python_files:  All non-code files found, categorised.
        file_type_summary: Aggregated count by extension, e.g. {'.md': 12, '.yml': 4}.
        language_summary:  Aggregated count by language, e.g. {'Python': 20, 'Go': 5}.
    """
    nodes: Dict[str, ModuleNode] = field(default_factory=dict)
    edges: List[tuple] = field(default_factory=list)
    uncovered_funcs: List[str] = field(default_factory=list)
    graph_json_path: Optional[str] = None
    non_python_files: List[NonPythonFile] = field(default_factory=list)
    file_type_summary: Dict[str, int] = field(default_factory=dict)
    language_summary: Dict[str, int] = field(default_factory=dict)


# ===========================================================================
# Stage 2 — Manifest / Dependency Models
# ===========================================================================

@dataclass
class ProjectMeta:
    """
    Structured output from manifest parsing — everything ContribTriage
    learns from the repository's config files across all ecosystems.

    Attributes:
        declared_deps:      Package names from all discovered manifests.
        dep_ecosystem:      Which ecosystem each dep belongs to:
                            {'requests': 'python', 'react': 'node', ...}
        test_framework:     Detected runners: ['pytest', 'npm test', 'cargo test']
        test_dirs:          Directories where test files were found.
        python_version_req: Required Python version string (e.g. '>=3.9').
        node_version_req:   Required Node version string (if present).
        repo_root:          Absolute path to the repository root.
        has_docker:         True if a Dockerfile or docker-compose.yml exists.
        ecosystems:         Active language ecosystems detected:
                            e.g. ['python', 'node', 'rust']

    Note on contributor guidelines:
        CONTRIBUTING.md is intentionally NOT stored here. Stage 1 already
        chunks and embeds it into Qdrant. The LLM retrieves relevant chunks
        on-demand via VectorStore.query() — no raw text bloat in the state.
    """
    declared_deps: List[str] = field(default_factory=list)
    dep_ecosystem: Dict[str, str] = field(default_factory=dict)
    test_framework: List[str] = field(default_factory=list)
    test_dirs: List[str] = field(default_factory=list)
    python_version_req: Optional[str] = None
    node_version_req: Optional[str] = None
    repo_root: str = ""
    has_docker: bool = False
    ecosystems: List[str] = field(default_factory=list)


# ===========================================================================
# Stage 3 — Environment Audit Models
# ===========================================================================

@dataclass
class SystemToolStatus:
    """
    Presence and resolution info for a single system-level tool.

    Attributes:
        name:           Tool name (e.g. 'postgres', 'redis', 'docker').
        found:          True if shutil.which() located the binary.
        path:           Resolved binary path if found.
        version:        Version string if detectable (e.g. '15.2').
        docker_snippet: Ready-to-run `docker run ...` command if Docker is
                        running and can substitute for the missing tool.
        os_install_cmd: OS-specific install command (brew/apt/winget).
        user_action:    Plain-English instruction if no auto-resolution exists.
    """
    name: str
    found: bool
    path: Optional[str] = None
    version: Optional[str] = None
    docker_snippet: Optional[str] = None
    os_install_cmd: Optional[str] = None
    user_action: Optional[str] = None


@dataclass
class EnvReport:
    """
    Full snapshot of the contributor's local environment.

    Attributes:
        os_name:          Platform string ('Windows', 'Linux', 'Darwin').
        os_version:       Detailed OS version string.
        python_version:   Active Python version (e.g. '3.11.4').
        python_path:      Absolute path to the Python executable.
        node_version:     Node.js version if found (e.g. '20.11.0').
        rust_version:     Rust/cargo version if found (e.g. '1.78.0').
        go_version:       Go version if found (e.g. '1.22.1').
        in_venv:          True if a virtualenv is active.
        venv_type:        'venv', 'conda', or None.
        venv_path:        Path to the active virtual environment.
        uv_available:     True if `uv` is on PATH.
        pip_available:    True if `pip` is on PATH.
        npm_available:    True if `npm` is on PATH.
        pnpm_available:   True if `pnpm` is on PATH.
        yarn_available:   True if `yarn` is on PATH.
        cargo_available:  True if `cargo` is on PATH.
        go_available:     True if `go` is on PATH.
        docker_running:   True if Docker daemon is active.
        system_tools:     Status of each checked system-level service tool.
    """
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    python_path: str = ""
    node_version: Optional[str] = None
    rust_version: Optional[str] = None
    go_version: Optional[str] = None
    in_venv: bool = False
    venv_type: Optional[str] = None
    venv_path: Optional[str] = None
    uv_available: bool = False
    pip_available: bool = False
    npm_available: bool = False
    pnpm_available: bool = False
    yarn_available: bool = False
    cargo_available: bool = False
    go_available: bool = False
    docker_running: bool = False
    system_tools: List[SystemToolStatus] = field(default_factory=list)


# ===========================================================================
# Dependency Resolution Models
# ===========================================================================

class DepStatus(enum.Enum):
    """Resolution status of a single dependency."""
    PRESENT    = "present"    # Already importable / installed
    INSTALLED  = "installed"  # Was missing; installer resolved it successfully
    FAILED     = "failed"     # Install attempted but failed
    DECLINED   = "declined"   # User answered [N] at the prompt
    SYSTEM_DEP = "system_dep" # Not a language package — OS-level tool


@dataclass
class DependencyFinding:
    """
    Resolution record for a single declared dependency.

    Attributes:
        name:        Package / tool name.
        dep_type:    'python' | 'node' | 'rust' | 'go' | 'system'
        status:      Final DepStatus after all resolution attempts.
        install_log: stdout/stderr from the installer subprocess.
        notes:       Extra context (Docker snippet, OS install command, etc.)
    """
    name: str
    dep_type: str
    status: DepStatus = DepStatus.PRESENT
    install_log: str = ""
    notes: str = ""


# ===========================================================================
# Verification Models
# ===========================================================================

class TestOutcome(enum.Enum):
    """Outcome classification of a single test item."""
    PASSED  = "passed"
    FAILED  = "failed"
    ERROR   = "error"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """
    Summary of a test run — real or generated, any ecosystem.

    Attributes:
        source:        'existing' | 'generated' (ContribTriage wrote the tests)
        ecosystem:     'python' | 'node' | 'rust' | 'go'
        passed:        Count of passing tests.
        failed:        Count of failing tests.
        errors:        Count of collection / compile errors.
        skipped:       Count of skipped tests.
        failed_items:  (test_id, short_error_msg) tuples — used for selective rerun.
        dep_errors:    Import/module errors extracted from output — triggers
                       the dependency healing loop in the orchestrator.
        code_bugs:     Failures Groq classified as pure code logic errors
                       (not env issues) — these become contribution suggestions.
        raw_output:    Full captured stdout + stderr.
        command_used:  The exact command list that was run.
    """
    source: str = "existing"
    ecosystem: str = "python"
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failed_items: List[tuple] = field(default_factory=list)
    dep_errors: List[str] = field(default_factory=list)
    code_bugs: List[str] = field(default_factory=list)
    raw_output: str = ""
    command_used: List[str] = field(default_factory=list)


# ===========================================================================
# LangGraph State — TypedDict (required by LangGraph's StateGraph)
# ===========================================================================

class LangGraphState(TypedDict, total=False):
    """
    The canonical state object that flows through every LangGraph node.

    Why TypedDict (not dataclass):
      LangGraph merges state between nodes using partial-dict updates.
      It cannot introspect dataclasses for this merge — TypedDict is required.
      The existing dataclasses above are nested as VALUES inside this dict.

    Annotated[list, operator.add] fields use the "append" reducer:
      When multiple nodes update the same key, the lists are concatenated
      rather than replaced. This is how terminal_log_history accumulates
      across every node in the pipeline without any node overwriting prior logs.

    Fields marked total=False are optional (may not be set at every stage).
    """

    # ── Inputs set by CLI (never mutated by nodes) ────────────────────────
    repo_path: str                          # Absolute path to cloned repo
    yes_flag: bool                          # --yes: auto-accept all prompts
    max_retries: int                        # --max-retries ceiling
    persist: bool                           # --persist: use SqliteSaver

    # ── LLM clients (injected by CLI, passed through state) ──────────────
    groq_client: Any                        # Initialised groq.Groq() instance

    # ── Stage 1 outputs ───────────────────────────────────────────────────
    knowledge_graph: KnowledgeGraph         # Populated by lexical_parser
    vector_store_path: str                  # Path to .contribtriage_qdrant/

    # ── Stage 2 outputs ───────────────────────────────────────────────────
    project_meta: ProjectMeta               # Manifests + test framework hints
    dep_findings: List[DependencyFinding]   # Grows across heal iterations

    # ── Stage 3 — env audit ───────────────────────────────────────────────
    env_report: EnvReport

    # ── LLM reasoning context (accumulates — append reducer) ─────────────
    terminal_log_history: Annotated[List[str], operator.add]
    groq_analysis: str                      # Latest Groq diagnosis text
    failure_category: str                   # 'code_bug'|'system_dep'|'app_dep'|''
    fix_command: str                        # LLM-generated shell command to run

    # ── Healing loop bookkeeping ──────────────────────────────────────────
    failed_test_ids: List[str]              # IDs for selective rerun
    retry_count: int                        # Incremented per healing attempt
    skipped_to_report: bool                 # True when user said N to system dep

    # ── Verification ──────────────────────────────────────────────────────
    test_result: Optional[TestResult]

    # ── Final output ──────────────────────────────────────────────────────
    final_report_path: str                  # Absolute path to SETUP_DIAGNOSTICS.md

