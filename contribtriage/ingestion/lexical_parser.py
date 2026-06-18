"""
contribtriage/ingestion/lexical_parser.py

Stage 1a — Universal Lexical Parser.

Instead of language-specific compiler toolchains, this module uses curated
regex signatures to extract structural maps across Python, JavaScript,
TypeScript, Rust, and Go. This keeps the parser dependency-free, fast,
and resilient — it never crashes due to a missing compiler or syntax error.

Extracted per file:
  - Class / struct / interface / enum definitions
  - Function / method definitions
  - Import / require / use / mod statements
  - TODO / FIXME / BUG / HACK inline comments

Output: a populated KnowledgeGraph dataclass backed by a NetworkX DiGraph.

Graph schema:
  V = {module nodes}     identified by dot-separated logical name
  E = {import edges}     directed: (importer → imported), labelled 'imports'
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from contribtriage.exceptions import IngestionError
from contribtriage.models import KnowledgeGraph, ModuleNode, NonPythonFile

console = Console()


# ===========================================================================
# Language Configuration
# ===========================================================================

@dataclass
class LanguageConfig:
    """Regex bundle for one programming language."""
    language: str
    class_pattern: re.Pattern
    function_pattern: re.Pattern
    import_patterns: List[re.Pattern]
    todo_pattern: re.Pattern


# Build the configs at module load — compiled once, reused for every file.
_TODO_PYTHON   = re.compile(r"#\s*(TODO|FIXME|BUG|HACK|XXX)[:\s]+(.*)", re.IGNORECASE)
_TODO_C_STYLE  = re.compile(r"//\s*(TODO|FIXME|BUG|HACK|XXX)[:\s]+(.*)", re.IGNORECASE)

LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {

    # ── Python ──────────────────────────────────────────────────────────
    ".py": LanguageConfig(
        language="Python",
        class_pattern=re.compile(
            r"^class\s+(\w+)", re.MULTILINE
        ),
        function_pattern=re.compile(
            r"^(?:async\s+)?def\s+(\w+)", re.MULTILINE
        ),
        import_patterns=[
            re.compile(r"^import\s+([\w.]+)", re.MULTILINE),
            re.compile(r"^from\s+([\w.]+)\s+import", re.MULTILINE),
        ],
        todo_pattern=_TODO_PYTHON,
    ),

    # ── JavaScript ──────────────────────────────────────────────────────
    ".js": LanguageConfig(
        language="JavaScript",
        class_pattern=re.compile(
            r"\bclass\s+(\w+)", re.MULTILINE
        ),
        function_pattern=re.compile(
            r"(?:^|\s)(?:export\s+)?(?:async\s+)?function\s+(\w+)"
            r"|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(",
            re.MULTILINE,
        ),
        import_patterns=[
            re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", re.MULTILINE),
            re.compile(r"""from\s+['"]([^'"]+)['"]""", re.MULTILINE),
            re.compile(r"""import\s+['"]([^'"]+)['"]""", re.MULTILINE),
        ],
        todo_pattern=_TODO_C_STYLE,
    ),

    # ── TypeScript (same as JS + extra patterns) ─────────────────────────
    ".ts": LanguageConfig(
        language="TypeScript",
        class_pattern=re.compile(
            r"\b(?:class|interface|type)\s+(\w+)", re.MULTILINE
        ),
        function_pattern=re.compile(
            r"(?:^|\s)(?:export\s+)?(?:async\s+)?function\s+(\w+)"
            r"|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(",
            re.MULTILINE,
        ),
        import_patterns=[
            re.compile(r"""from\s+['"]([^'"]+)['"]""", re.MULTILINE),
            re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", re.MULTILINE),
            re.compile(r"""import\s+type\s+.*\s+from\s+['"]([^'"]+)['"]""", re.MULTILINE),
        ],
        todo_pattern=_TODO_C_STYLE,
    ),

    # ── TSX / JSX (React components — same structural patterns) ──────────
    ".tsx": None,  # filled below by reference
    ".jsx": None,

    # ── Rust ─────────────────────────────────────────────────────────────
    ".rs": LanguageConfig(
        language="Rust",
        class_pattern=re.compile(
            r"^(?:pub(?:\s*\([^)]*\))?\s+)?(?:struct|enum|trait|impl)\s+(\w+)",
            re.MULTILINE,
        ),
        function_pattern=re.compile(
            # No ^ anchor — impl block methods are indented, not at column 0
            r"(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)",
            re.MULTILINE,
        ),
        import_patterns=[
            re.compile(r"^use\s+([\w:]+(?:::\{[^}]+\})?)", re.MULTILINE),
            re.compile(r"^extern\s+crate\s+(\w+)", re.MULTILINE),
            re.compile(r"^mod\s+(\w+)", re.MULTILINE),
        ],
        todo_pattern=_TODO_C_STYLE,
    ),

    # ── Go ───────────────────────────────────────────────────────────────
    ".go": LanguageConfig(
        language="Go",
        class_pattern=re.compile(
            r"^type\s+(\w+)\s+(?:struct|interface)", re.MULTILINE
        ),
        function_pattern=re.compile(
            r"^func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s+)?(\w+)", re.MULTILINE
        ),
        import_patterns=[
            # matches both single `import "pkg"` and block `import (\n"pkg"\n)`
            re.compile(r'"([\w./\-]+)"', re.MULTILINE),
        ],
        todo_pattern=_TODO_C_STYLE,
    ),
}

# Fill aliases
LANGUAGE_CONFIGS[".tsx"] = LanguageConfig(
    language="TypeScript/React",
    class_pattern=LANGUAGE_CONFIGS[".ts"].class_pattern,
    function_pattern=LANGUAGE_CONFIGS[".ts"].function_pattern,
    import_patterns=LANGUAGE_CONFIGS[".ts"].import_patterns,
    todo_pattern=_TODO_C_STYLE,
)
LANGUAGE_CONFIGS[".jsx"] = LanguageConfig(
    language="JavaScript/React",
    class_pattern=LANGUAGE_CONFIGS[".js"].class_pattern,
    function_pattern=LANGUAGE_CONFIGS[".js"].function_pattern,
    import_patterns=LANGUAGE_CONFIGS[".js"].import_patterns,
    todo_pattern=_TODO_C_STYLE,
)

# Extensions that are code files (get structurally parsed)
CODE_EXTENSIONS: Set[str] = set(LANGUAGE_CONFIGS.keys())

# Non-code file categorization map  (extension → category label)
_EXT_CATEGORY: Dict[str, str] = {
    ".md":   "docs",   ".rst":  "docs",  ".txt": "docs",
    ".yml":  "ci",     ".yaml": "ci",
    ".json": "config", ".toml": "config", ".ini": "config",
    ".cfg":  "config", ".env":  "config",
    ".sh":   "config", ".bash": "config", ".zsh": "config",
    ".css":  "frontend", ".scss": "frontend", ".html": "frontend",
    ".csv":  "data",   ".xml":  "data",
}
_DOCKER_NAMES = {"dockerfile", "docker-compose.yml", "docker-compose.yaml"}

# Directories to skip entirely
_SKIP_DIRS: Set[str] = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", "env", ".env",
    "dist", "build", ".next", ".nuxt",
    "site-packages", ".contribtriage_qdrant",
}


# ===========================================================================
# Public API
# ===========================================================================

def build_knowledge_graph(
    repo_root: str,
    output_dir: Optional[str] = None,
) -> KnowledgeGraph:
    """
    Parse every supported source file under *repo_root* and return a
    populated KnowledgeGraph.

    Args:
        repo_root:   Absolute path to the cloned repository.
        output_dir:  Where to write graph.json. Defaults to
                     ``<repo_root>/.contribtriage/``

    Returns:
        KnowledgeGraph with nodes, edges, uncovered_funcs, non_python_files,
        file_type_summary, language_summary, and graph_json_path set.
    """
    root = Path(repo_root).resolve()
    out_dir = Path(output_dir) if output_dir else root / ".contribtriage"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = _collect_all_files(root)
    code_files = [f for f in all_files if f.suffix in CODE_EXTENSIONS]
    other_files = [f for f in all_files if f.suffix not in CODE_EXTENSIONS]

    console.print(
        f"[cyan]  → {len(code_files)} code files, "
        f"{len(other_files)} other files found[/cyan]"
    )

    graph = nx.DiGraph()
    nodes: Dict[str, ModuleNode] = {}
    ingest_errors: List[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Parsing source files...", total=len(code_files))

        for f in code_files:
            progress.advance(task)
            cfg = LANGUAGE_CONFIGS.get(f.suffix)
            if cfg is None:
                continue
            module_name = _path_to_logical_name(f, root)
            try:
                node = _parse_file(f, module_name, cfg)
            except IngestionError as exc:
                ingest_errors.append(str(exc))
                console.print(
                    f"[yellow]  ⚠ Skipped: {f.name} — {exc}[/yellow]"
                )
                continue
            nodes[module_name] = node
            graph.add_node(module_name, **_node_to_attrs(node))

    # Build directed import edges between known project modules
    edges = _build_import_edges(nodes, graph)

    # Classify non-code files
    non_python: List[NonPythonFile] = [
        _classify_non_code(f) for f in other_files
    ]

    # Cross-ref test modules to find uncovered public functions
    test_modules = {n for n in nodes if "test" in n.lower()}
    uncovered = _find_uncovered(nodes, test_modules)

    # Aggregate stats
    file_type_summary = _count_by_ext(all_files)
    language_summary  = _count_by_language(nodes)

    # Serialize graph
    graph_json_path = out_dir / "graph.json"
    _write_graph(graph, graph_json_path)

    if ingest_errors:
        console.print(
            f"[yellow]  ⚠ {len(ingest_errors)} file(s) skipped during parsing[/yellow]"
        )

    console.print(
        f"[green]  ✓ Knowledge graph: {len(nodes)} modules, "
        f"{len(edges)} import edges, {len(uncovered)} uncovered functions[/green]"
    )

    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        uncovered_funcs=uncovered,
        graph_json_path=str(graph_json_path),
        non_python_files=non_python,
        file_type_summary=file_type_summary,
        language_summary=language_summary,
    )


# ===========================================================================
# File Collection
# ===========================================================================

def _collect_all_files(root: Path) -> List[Path]:
    """Return all files under root, skipping ignored directories."""
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Skip if any parent dir component is in the skip list
        if any(
            part in _SKIP_DIRS or part.endswith(".egg-info")
            for part in path.parts
        ):
            continue
        files.append(path)
    return sorted(files)


def _path_to_logical_name(file: Path, root: Path) -> str:
    """
    Convert an absolute file path to a unique dot-separated logical name.

    Python convention: drop the .py extension and collapse __init__.
      src/utils.py      → 'src.utils'
      src/__init__.py   → 'src'

    Non-Python: keep the extension as an underscore suffix to guarantee
    uniqueness in polyglot repos where the same stem may exist in multiple
    languages (e.g. main.py + main.go in the same directory).
      src/utils.ts      → 'src.utils_ts'
      src/main.go       → 'src.main_go'
      src/lib.rs        → 'src.lib_rs'
    """
    rel = file.relative_to(root)
    suffix = file.suffix  # e.g. '.py', '.ts', '.rs'

    if suffix == ".py":
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
    else:
        # Include extension as underscore suffix on the final segment
        stem_parts = list(rel.with_suffix("").parts)
        if stem_parts:
            ext_tag = suffix.lstrip(".")       # '.ts' → 'ts'
            stem_parts[-1] = f"{stem_parts[-1]}_{ext_tag}"
        parts = stem_parts

    return ".".join(parts) if parts else (file.stem + "_" + suffix.lstrip("."))


# ===========================================================================
# Per-File Parsing
# ===========================================================================

def _parse_file(
    file: Path,
    module_name: str,
    cfg: LanguageConfig,
) -> ModuleNode:
    """
    Apply regex patterns for *cfg* to *file* and return a ModuleNode.

    Raises:
        IngestionError: if the file cannot be read (OS error).
                        SyntaxErrors never occur — we use regex, not AST.
    """
    try:
        source = file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise IngestionError(str(file), exc) from exc

    classes   = _extract(cfg.class_pattern, source)
    functions = _extract(cfg.function_pattern, source)
    imports   = _extract_imports(cfg.import_patterns, source)
    todos     = _extract_todos(cfg.todo_pattern, source)

    # Best-effort docstring: first block comment or triple-quoted string
    docstring = _extract_docstring(source, cfg.language)

    return ModuleNode(
        path=str(file),
        module_name=module_name,
        language=cfg.language,
        classes=classes,
        functions=functions,
        imports=imports,
        todos=todos,
        docstring=docstring,
    )


def _extract(pattern: re.Pattern, source: str) -> List[str]:
    """Find all non-empty capture groups from a pattern."""
    results = []
    for match in pattern.finditer(source):
        # Some patterns have multiple groups (e.g. JS function alternatives)
        name = next((g for g in match.groups() if g), None)
        if name:
            results.append(name)
    return list(dict.fromkeys(results))  # deduplicate, preserve order


def _extract_imports(patterns: List[re.Pattern], source: str) -> List[str]:
    """Collect all imported module / package names, deduplicated."""
    seen: Set[str] = set()
    imports: List[str] = []
    for pattern in patterns:
        for match in pattern.finditer(source):
            name = next((g for g in match.groups() if g), None)
            if name and name not in seen:
                seen.add(name)
                # Normalise: take top-level name only
                # Handles: 'os.path' → 'os'  (Python dot-separated)
                #           'std::fs'  → 'std' (Rust :: separated)
                #           'some/pkg' → 'some' (Go slash-separated)
                imports.append(name.split("::")[0].split(".")[0].split("/")[0])
    return imports


def _extract_todos(pattern: re.Pattern, source: str) -> List[str]:
    """Extract TODO / FIXME / BUG / HACK comment text."""
    todos = []
    for match in pattern.finditer(source):
        tag  = match.group(1).upper()
        text = match.group(2).strip()
        todos.append(f"{tag}: {text}")
    return todos


def _extract_docstring(source: str, language: str) -> Optional[str]:
    """
    Best-effort extraction of the file's top-level documentation comment.
    Returns the first triple-quoted string (Python) or block comment (JS/Rust/Go).
    """
    if language == "Python":
        m = re.search(r'"""(.*?)"""', source[:500], re.DOTALL)
        if m:
            return m.group(1).strip()[:300]
    else:
        # C-style block comment: /* ... */
        m = re.search(r"/\*(.*?)\*/", source[:500], re.DOTALL)
        if m:
            return m.group(1).strip()[:300]
    return None


# ===========================================================================
# Graph Construction
# ===========================================================================

def _build_import_edges(
    nodes: Dict[str, ModuleNode],
    graph: nx.DiGraph,
) -> List[tuple]:
    """
    Add directed 'imports' edges to the graph for intra-project relationships.
    Third-party / stdlib imports are recorded in ModuleNode.imports but are
    NOT added as graph edges (those are handled by the dependency resolver).
    """
    edges: List[tuple] = []
    known: Set[str] = set(nodes.keys())

    for src_name, node in nodes.items():
        for imported in node.imports:
            # Match project-internal modules by exact name or prefix
            targets = [
                m for m in known
                if m == imported or m.startswith(imported + ".")
            ]
            for tgt in targets:
                if not graph.has_edge(src_name, tgt):
                    graph.add_edge(src_name, tgt, relation="imports")
                    edges.append((src_name, tgt, "imports"))

    return edges


def _node_to_attrs(node: ModuleNode) -> dict:
    """Flatten a ModuleNode to simple NetworkX node attributes."""
    return {
        "path":       node.path,
        "language":   node.language,
        "classes":    node.classes,
        "functions":  node.functions,
        "todo_count": len(node.todos),
    }


# ===========================================================================
# Coverage Analysis
# ===========================================================================

def _find_uncovered(
    nodes: Dict[str, ModuleNode],
    test_modules: Set[str],
) -> List[str]:
    """
    Return 'module.function' strings for public functions that are never
    referenced in any test module.

    Strategy: collect every identifier present in test file source text
    via a broad regex scan (function names, variable names, etc.).
    Any public function NOT appearing there is flagged as uncovered.
    """
    referenced: Set[str] = set()
    for t_name in test_modules:
        node = nodes[t_name]
        referenced.update(node.functions)
        try:
            src = Path(node.path).read_text(encoding="utf-8", errors="replace")
            referenced.update(re.findall(r"\b([a-zA-Z_]\w*)\b", src))
        except OSError:
            pass

    uncovered: List[str] = []
    for mod_name, node in nodes.items():
        if "test" in mod_name.lower():
            continue
        for func in node.functions:
            if func.startswith("_"):
                continue  # private — not expected to have direct test coverage
            if func not in referenced:
                uncovered.append(f"{mod_name}.{func}")

    return uncovered


# ===========================================================================
# Non-Code File Classification
# ===========================================================================

def _classify_non_code(file: Path) -> NonPythonFile:
    """Categorise a non-code file into a human-readable bucket."""
    name_lower = file.name.lower()
    ext = file.suffix.lower()
    size_kb = round(file.stat().st_size / 1024, 2)

    if name_lower in _DOCKER_NAMES or name_lower.startswith("dockerfile"):
        category = "docker"
    else:
        category = _EXT_CATEGORY.get(ext, "other")

    return NonPythonFile(
        path=str(file),
        ext=ext or file.name,  # for extension-less files like 'Dockerfile'
        category=category,
        size_kb=size_kb,
    )


# ===========================================================================
# Aggregation Helpers
# ===========================================================================

def _count_by_ext(files: List[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for f in files:
        key = f.suffix or f.name
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_by_language(nodes: Dict[str, ModuleNode]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in nodes.values():
        counts[node.language] = counts.get(node.language, 0) + 1
    return counts


# ===========================================================================
# Graph Serialization
# ===========================================================================

def _write_graph(graph: nx.DiGraph, output_path: Path) -> None:
    """Serialize the NetworkX graph to JSON node-link format."""
    data = nx.node_link_data(graph)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
