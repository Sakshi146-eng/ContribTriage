"""
contribtriage/ingestion/lexical_parser.py

Stage 1a — Universal Lexical Parser (Tree-sitter edition).

Uses Tree-sitter ASTs instead of regexes for precise, language-native extraction
across Python, JavaScript/JSX, TypeScript/TSX, Rust, and Go.

Tree-sitter 0.23+ API (modern precompiled wheels):
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    lang = Language(tspython.language())
    parser = Parser(lang)

Query captures() returns dict[str, list[Node]] in 0.23+.

Extracted per file:
  - Class / struct / interface / enum / trait definitions
  - Function / method / arrow-function definitions
  - Import / use / require statements (as raw text, normalised in Python)
  - TODO / FIXME / BUG / HACK inline comments

Output: a populated KnowledgeGraph backed by NetworkX DiGraph.

Graph schema:
  V = {module nodes}   identified by dot-separated logical name
  E = {import edges}   directed: (importer → imported), labelled 'imports'
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

# ── Tree-sitter 0.23 modern API ───────────────────────────────────────────
import tree_sitter_go        as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_python    as tspy
import tree_sitter_rust      as tsrs
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

from contribtriage.exceptions import IngestionError
from contribtriage.models import KnowledgeGraph, ModuleNode, NonPythonFile

console = Console(force_terminal=True, highlight=False)

# ===========================================================================
# Language initialisation (once at import time — zero overhead per file)
# ===========================================================================

_LANG_PY  = Language(tspy.language())
_LANG_JS  = Language(tsjs.language())
_LANG_TS  = Language(tsts.language_typescript())
_LANG_TSX = Language(tsts.language_tsx())
_LANG_GO  = Language(tsgo.language())
_LANG_RS  = Language(tsrs.language())


# ===========================================================================
# Language Configuration (Tree-sitter edition)
# ===========================================================================

@dataclass
class LanguageConfig:
    """Tree-sitter query bundle for one programming language."""
    language: str
    ts_language: Language       # Compiled Language object
    func_queries: List[str]     # S-expression query strings → capture @name
    class_queries: List[str]    # S-expression query strings → capture @name
    import_stmts_query: str     # S-expression query string → capture @stmt (full text)
    import_src_query: str       # S-expression query string → capture @src (source value only)
    todo_pattern: re.Pattern    # Regex for TODO/FIXME comments (language-agnostic)


_TODO_PY    = re.compile(r"#\s*(TODO|FIXME|BUG|HACK|XXX)[:\s]+(.*)", re.IGNORECASE)
_TODO_C     = re.compile(r"//\s*(TODO|FIXME|BUG|HACK|XXX)[:\s]+(.*)", re.IGNORECASE)

LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {

    # ── Python ──────────────────────────────────────────────────────────
    ".py": LanguageConfig(
        language="Python",
        ts_language=_LANG_PY,
        func_queries=[
            "(function_definition name: (identifier) @name)",
        ],
        class_queries=[
            "(class_definition name: (identifier) @name)",
        ],
        import_stmts_query="[(import_statement) (import_from_statement)] @stmt",
        import_src_query="",   # parsed from full stmt text
        todo_pattern=_TODO_PY,
    ),

    # ── JavaScript / JSX ────────────────────────────────────────────────
    ".js": LanguageConfig(
        language="JavaScript",
        ts_language=_LANG_JS,
        func_queries=[
            "(function_declaration name: (identifier) @name)",
            "(method_definition name: (property_identifier) @name)",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function)))",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression)))",
            "(variable_declaration (variable_declarator name: (identifier) @name value: (arrow_function)))",
        ],
        class_queries=[
            "(class_declaration name: (identifier) @name)",
        ],
        import_stmts_query="(import_statement) @stmt",
        import_src_query='(import_statement source: (string (string_fragment) @src))',
        todo_pattern=_TODO_C,
    ),

    # ── TypeScript ──────────────────────────────────────────────────────
    ".ts": LanguageConfig(
        language="TypeScript",
        ts_language=_LANG_TS,
        func_queries=[
            "(function_declaration name: (identifier) @name)",
            "(method_definition name: (property_identifier) @name)",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function)))",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression)))",
        ],
        class_queries=[
            "(class_declaration name: (type_identifier) @name)",
            "(interface_declaration name: (type_identifier) @name)",
            "(type_alias_declaration name: (type_identifier) @name)",
        ],
        import_stmts_query="(import_statement) @stmt",
        import_src_query='(import_statement source: (string (string_fragment) @src))',
        todo_pattern=_TODO_C,
    ),

    # ── TSX / JSX (React) ───────────────────────────────────────────────
    ".tsx": LanguageConfig(
        language="TypeScript/React",
        ts_language=_LANG_TSX,
        func_queries=[
            "(function_declaration name: (identifier) @name)",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function)))",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression)))",
        ],
        class_queries=[
            "(class_declaration name: (type_identifier) @name)",
            "(interface_declaration name: (type_identifier) @name)",
        ],
        import_stmts_query="(import_statement) @stmt",
        import_src_query='(import_statement source: (string (string_fragment) @src))',
        todo_pattern=_TODO_C,
    ),
    ".jsx": LanguageConfig(
        language="JavaScript/React",
        ts_language=_LANG_JS,
        func_queries=[
            "(function_declaration name: (identifier) @name)",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function)))",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression)))",
        ],
        class_queries=[
            "(class_declaration name: (identifier) @name)",
        ],
        import_stmts_query="(import_statement) @stmt",
        import_src_query='(import_statement source: (string (string_fragment) @src))',
        todo_pattern=_TODO_C,
    ),

    # ── Rust ─────────────────────────────────────────────────────────────
    ".rs": LanguageConfig(
        language="Rust",
        ts_language=_LANG_RS,
        func_queries=[
            "(function_item name: (identifier) @name)",
        ],
        class_queries=[
            "(struct_item name: (type_identifier) @name)",
            "(enum_item name: (type_identifier) @name)",
            "(trait_item name: (type_identifier) @name)",
            "(impl_item type: (type_identifier) @name)",
        ],
        import_stmts_query="(use_declaration) @stmt",
        import_src_query="",   # parsed from use text
        todo_pattern=_TODO_C,
    ),

    # ── Go ───────────────────────────────────────────────────────────────
    ".go": LanguageConfig(
        language="Go",
        ts_language=_LANG_GO,
        func_queries=[
            "(function_declaration name: (identifier) @name)",
            "(method_declaration name: (field_identifier) @name)",
        ],
        class_queries=[
            "(type_spec name: (type_identifier) @name)",
        ],
        import_stmts_query="",  # handled by import_src_query directly
        import_src_query="(import_spec path: (interpreted_string_literal) @src)",
        todo_pattern=_TODO_C,
    ),
}

# Extensions that are structured code files
CODE_EXTENSIONS: Set[str] = set(LANGUAGE_CONFIGS.keys())

# Non-code file categorization
_EXT_CATEGORY: Dict[str, str] = {
    ".md":   "docs",  ".rst":  "docs",  ".txt": "docs",
    ".yml":  "ci",    ".yaml": "ci",
    ".json": "config", ".toml": "config", ".ini": "config",
    ".cfg":  "config", ".env":  "config",
    ".sh":   "config", ".bash": "config", ".zsh": "config",
    ".css":  "frontend", ".scss": "frontend", ".html": "frontend",
    ".csv":  "data",  ".xml":  "data",
}
_DOCKER_NAMES = {"dockerfile", "docker-compose.yml", "docker-compose.yaml"}

# Directories to skip entirely
_SKIP_DIRS: Set[str] = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", "env", ".env",
    "dist", "build", ".next", ".nuxt",
    "site-packages", ".contribtriage_qdrant", ".contribtriage",
}


# ===========================================================================
# Public API
# ===========================================================================

def build_knowledge_graph(
    repo_root: str,
    output_dir: Optional[str] = None,
) -> KnowledgeGraph:
    """
    Parse every supported source file under *repo_root* using Tree-sitter
    and return a populated KnowledgeGraph.

    Args:
        repo_root:   Absolute path to the cloned repository.
        output_dir:  Where to write graph.json. Defaults to
                     ``<repo_root>/.contribtriage/``

    Returns:
        KnowledgeGraph with nodes, edges, uncovered_funcs, non_python_files,
        file_type_summary, language_summary, and graph_json_path set.
    """
    root    = Path(repo_root).resolve()
    out_dir = Path(output_dir) if output_dir else root / ".contribtriage"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files  = _collect_all_files(root)
    code_files = [f for f in all_files if f.suffix in CODE_EXTENSIONS]
    other_files = [f for f in all_files if f.suffix not in CODE_EXTENSIONS]

    console.print(
        f"[cyan]  -> {len(code_files)} code files, "
        f"{len(other_files)} other files found[/cyan]"
    )

    graph         = nx.DiGraph()
    nodes: Dict[str, ModuleNode] = {}
    ingest_errors: List[str] = []

    with Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Parsing source files (Tree-sitter)...", total=len(code_files))

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
                console.print(f"[yellow]  [!] Skipped: {f.name} — {exc}[/yellow]")
                continue
            nodes[module_name] = node
            graph.add_node(module_name, **_node_to_attrs(node))

    # Build directed import edges with relative-path resolution
    edges = _build_import_edges(nodes, graph, root)

    # Classify non-code files
    non_python: List[NonPythonFile] = [_classify_non_code(f) for f in other_files]

    # Cross-ref test modules → find uncovered public functions (all languages)
    test_modules = {n for n in nodes if "test" in n.lower()}
    uncovered    = _find_uncovered(nodes, test_modules)

    # Aggregate stats
    file_type_summary = _count_by_ext(all_files)
    language_summary  = _count_by_language(nodes)

    # Serialise graph
    graph_json_path = out_dir / "graph.json"
    _write_graph(graph, graph_json_path)

    if ingest_errors:
        console.print(
            f"[yellow]  [!] {len(ingest_errors)} file(s) skipped during parsing[/yellow]"
        )

    console.print(
        f"[green]  [OK] Knowledge graph: {len(nodes)} modules, "
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

    Python convention: drop .py extension and collapse __init__.
      src/utils.py      → 'src.utils'
      src/__init__.py   → 'src'

    Non-Python: keep the extension as an underscore suffix to guarantee
    uniqueness in polyglot repos.
      src/utils.ts      → 'src.utils_ts'
      src/main.go       → 'src.main_go'
      src/lib.rs        → 'src.lib_rs'
    """
    rel    = file.relative_to(root)
    suffix = file.suffix

    if suffix == ".py":
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
    else:
        stem_parts = list(rel.with_suffix("").parts)
        if stem_parts:
            ext_tag = suffix.lstrip(".")
            stem_parts[-1] = f"{stem_parts[-1]}_{ext_tag}"
        parts = stem_parts

    return ".".join(parts) if parts else (file.stem + "_" + suffix.lstrip("."))


# ===========================================================================
# Per-File Parsing (Tree-sitter)
# ===========================================================================

def _parse_file(
    file: Path,
    module_name: str,
    cfg: LanguageConfig,
) -> ModuleNode:
    """
    Parse a single source file with Tree-sitter and return a ModuleNode.

    Raises:
        IngestionError: if the file cannot be read (OS error).
    """
    try:
        source_bytes = file.read_bytes()
    except OSError as exc:
        raise IngestionError(str(file), exc) from exc

    source_text = source_bytes.decode("utf-8", errors="replace")

    parser = Parser(cfg.ts_language)
    tree   = parser.parse(source_bytes)

    functions = _query_names(cfg.ts_language, tree, cfg.func_queries)
    classes   = _query_names(cfg.ts_language, tree, cfg.class_queries)
    imports   = _extract_imports(cfg, tree, source_text)
    todos     = _extract_todos(cfg.todo_pattern, source_text)
    docstring = _extract_docstring(source_bytes, cfg.language)

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


def _query_names(
    language: Language,
    tree,
    query_strings: List[str],
) -> List[str]:
    """
    Run one or more Tree-sitter queries and collect all @name capture texts.
    Returns deduplicated list preserving order.
    """
    results: List[str] = []
    seen: Set[str] = set()
    for qs in query_strings:
        if not qs.strip():
            continue
        try:
            q = language.query(qs)
            for node in q.captures(tree.root_node).get("name", []):
                text = node.text.decode("utf-8", errors="replace")
                if text and text not in seen:
                    seen.add(text)
                    results.append(text)
        except Exception:  # noqa: BLE001 — malformed query on unusual source
            pass
    return results


def _extract_imports(
    cfg: LanguageConfig,
    tree,
    source_text: str,
) -> List[str]:
    """
    Extract and normalise import/require/use identifiers.

    Returns a list of top-level import names (e.g. 'os', 'react', 'std').
    """
    imports: List[str] = []
    seen: Set[str] = set()

    def _add(name: str) -> None:
        # Normalise to top-level: 'os.path' → 'os', 'std::fs' → 'std', './utils' → 'utils'
        name = name.strip().strip("'\"")
        if not name:
            return
        # Keep full path for relative JS/TS imports (resolved in edge builder)
        if name.startswith("."):
            root_name = name   # keep relative path intact
        else:
            root_name = (
                name.split("::")[0]
                    .split(".")[0]
                    .split("/")[-1]   # Go: take last segment for stdlib
                    .split(" ")[0]    # safety trim
            )
        if root_name and root_name not in seen:
            seen.add(root_name)
            imports.append(root_name)

    # ── Use language-specific source query if available ──────────────────
    if cfg.import_src_query:
        try:
            q = cfg.ts_language.query(cfg.import_src_query)
            for node in q.captures(tree.root_node).get("src", []):
                text = node.text.decode("utf-8", errors="replace").strip('"\'')
                _add(text)
        except Exception:  # noqa: BLE001
            pass

    # ── Fall back to full statement text ─────────────────────────────────
    elif cfg.import_stmts_query:
        try:
            q = cfg.ts_language.query(cfg.import_stmts_query)
            for node in q.captures(tree.root_node).get("stmt", []):
                stmt = node.text.decode("utf-8", errors="replace")
                _parse_import_stmt(stmt, cfg.language, seen, imports)
        except Exception:  # noqa: BLE001
            pass

    # ── Go: import_src_query only ─────────────────────────────────────────
    #    (handled above already — Go only uses import_src_query)

    # ── Rust: also catch extern crate declarations ─────────────────────────
    if cfg.language == "Rust":
        try:
            q = cfg.ts_language.query("(extern_crate_declaration name: (identifier) @name)")
            for node in q.captures(tree.root_node).get("name", []):
                text = node.text.decode("utf-8", errors="replace")
                _add(text)
        except Exception:  # noqa: BLE001
            pass

    # ── JS/TS: also catch require() calls ─────────────────────────────────
    if cfg.language in ("JavaScript", "JavaScript/React", "TypeScript", "TypeScript/React"):
        for m in re.finditer(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", source_text):
            _add(m.group(1))

    return imports


def _parse_import_stmt(stmt: str, language: str, seen: Set[str], imports: List[str]) -> None:
    """Parse the raw text of an import statement to extract module names."""
    stmt = stmt.strip()

    if language == "Python":
        # "import os.path" → "os"
        # "from contribtriage.models import X" → "contribtriage"
        if stmt.startswith("from "):
            parts = stmt.split()
            if len(parts) >= 2:
                mod = parts[1].split(".")[0]
                if mod and mod not in seen:
                    seen.add(mod)
                    imports.append(mod)
        elif stmt.startswith("import "):
            # "import os, sys" → ["os", "sys"]
            rest = stmt[len("import "):].strip()
            for part in rest.split(","):
                mod = part.strip().split(".")[0].split(" ")[0]
                if mod and mod not in seen:
                    seen.add(mod)
                    imports.append(mod)

    elif language in ("Rust",):
        # "use std::collections::HashMap;" → "std"
        # "use crate::models::X;" → "crate" (we keep "crate" as-is for edge matching)
        # NOTE: str.lstrip("use ") strips individual chars, NOT the prefix. Use slice instead.
        if stmt.startswith("use "):
            rest = stmt[4:].rstrip(";").strip()
        else:
            rest = stmt.rstrip(";").strip()
        root = rest.split("::")[0].strip()
        if root and root not in seen:
            seen.add(root)
            imports.append(root)


def _extract_todos(pattern: re.Pattern, source_text: str) -> List[str]:
    """Extract TODO / FIXME / BUG / HACK comment text."""
    todos = []
    for m in pattern.finditer(source_text):
        tag  = m.group(1).upper()
        text = m.group(2).strip()
        todos.append(f"{tag}: {text}")
    return todos


def _extract_docstring(source_bytes: bytes, language: str) -> Optional[str]:
    """
    Best-effort extraction of the file's top-level documentation comment.

    Python: first triple-quoted string.
    Others: first block comment (/* ... */) or line comment (// ...) block.
    """
    source_head = source_bytes[:600].decode("utf-8", errors="replace")
    if language == "Python":
        m = re.search(r'"""(.*?)"""', source_head, re.DOTALL)
        if m:
            return m.group(1).strip()[:300]
        m = re.search(r"'''(.*?)'''", source_head, re.DOTALL)
        if m:
            return m.group(1).strip()[:300]
    else:
        m = re.search(r"/\*(.*?)\*/", source_head, re.DOTALL)
        if m:
            return m.group(1).strip()[:300]
    return None


# ===========================================================================
# Graph Construction
# ===========================================================================

def _build_import_edges(
    nodes: Dict[str, ModuleNode],
    graph: nx.DiGraph,
    repo_root: Path,
) -> List[tuple]:
    """
    Add directed 'imports' edges to the graph for intra-project relationships.

    Supports:
    - Python: dot-separated module name prefix matching
    - JS/TS/JSX/TSX: relative path resolution (./utils → actual file)
    - Rust: `crate::` prefix matching
    - Go: package path last-segment matching
    """
    edges: List[tuple] = []

    # Build reverse map: absolute file path → logical module name
    path_to_logical: Dict[Path, str] = {
        Path(node.path).resolve(): name
        for name, node in nodes.items()
    }
    known_names: Set[str] = set(nodes.keys())

    for src_name, node in nodes.items():
        src_file = Path(node.path).resolve()
        src_dir  = src_file.parent
        language = node.language

        for imported in node.imports:
            targets = _resolve_import(
                imported, language, src_dir, repo_root,
                path_to_logical, known_names,
            )
            for tgt in targets:
                if tgt != src_name and not graph.has_edge(src_name, tgt):
                    graph.add_edge(src_name, tgt, relation="imports")
                    edges.append((src_name, tgt, "imports"))

    return edges


def _resolve_import(
    imported: str,
    language: str,
    src_dir: Path,
    repo_root: Path,
    path_to_logical: Dict[Path, str],
    known_names: Set[str],
) -> List[str]:
    """
    Resolve a raw import string to known module names in this project.
    Third-party / stdlib imports that don't match anything are ignored.
    """
    # ── JS / TS / JSX / TSX: resolve relative paths ──────────────────────
    if language in (
        "JavaScript", "JavaScript/React",
        "TypeScript", "TypeScript/React",
    ):
        if imported.startswith("."):
            # Relative import: try with each code extension
            base = (src_dir / imported).resolve()
            for ext in (".jsx", ".tsx", ".js", ".ts", ""):
                candidate = base.with_suffix(ext) if ext else base
                if candidate in path_to_logical:
                    return [path_to_logical[candidate]]
                # Also try /index variants
                index_candidate = base / f"index{ext}"
                if index_candidate in path_to_logical:
                    return [path_to_logical[index_candidate]]
            return []
        else:
            # Bare npm package: check if it matches an internal module stem
            stem = imported.replace("-", "_").replace("/", "_")
            return [
                n for n in known_names
                if n.split(".")[-1].replace("_jsx", "").replace("_tsx", "")
                   .replace("_js", "").replace("_ts", "") == stem
            ]

    # ── Python: dot-path prefix matching ─────────────────────────────────
    elif language == "Python":
        return [
            n for n in known_names
            if n == imported or n.startswith(imported + ".")
        ]

    # ── Rust: crate:: → match within-repo modules ─────────────────────────
    elif language == "Rust":
        if imported in ("crate", "self", "super"):
            return []   # self-references — not useful as graph edges
        if imported == "std" or imported == "core" or imported == "alloc":
            return []   # stdlib
        # Match external crate names against repo module names
        return [
            n for n in known_names
            if n.split(".")[-1].replace("_rs", "").replace("_lib", "") == imported
        ]

    # ── Go: last path segment matching ───────────────────────────────────
    elif language == "Go":
        last_seg = imported.split("/")[-1]
        return [
            n for n in known_names
            if n.split(".")[-1].replace("_go", "") == last_seg
        ]

    return []


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
    Return 'module.function' strings for public functions that have NO
    corresponding test function following the ecosystem naming convention.

    Convention per language
    -----------------------
      Python / Rust / JS / TS:
          source function   ``validate_schema``
          expected test     ``test_validate_schema``

      Go:
          source function   ``RunPipeline``
          expected test     ``TestRunPipeline``

    A function is "covered" when ANY test module contains a function
    whose name matches the pattern above.

    This is far more precise than raw token-scanning: a string literal or
    comment containing the word "login" does NOT count as covering a
    ``login()`` function.
    """
    # Collect all test-function names declared in every test module
    all_test_funcs: Set[str] = set()
    for t_name in test_modules:
        all_test_funcs.update(nodes[t_name].functions)

    uncovered: List[str] = []
    for mod_name, node in nodes.items():
        if "test" in mod_name.lower():
            continue
        language = node.language
        for func in node.functions:
            if func.startswith("_"):
                continue   # private — not expected to have a dedicated test

            # Build the expected test-function name for this language
            if language == "Go":
                # Go convention: TestFuncName  (capitalise first letter)
                expected = f"Test{func[0].upper()}{func[1:]}" if func else ""
            else:
                # Python / Rust / JS / TS convention: test_func_name
                expected = f"test_{func}"

            if expected not in all_test_funcs:
                uncovered.append(f"{mod_name}.{func}")

    return uncovered



# ===========================================================================
# Non-Code File Classification
# ===========================================================================

def _classify_non_code(file: Path) -> NonPythonFile:
    """Categorise a non-code file into a human-readable bucket."""
    name_lower = file.name.lower()
    ext        = file.suffix.lower()
    size_kb    = round(file.stat().st_size / 1024, 2)

    if name_lower in _DOCKER_NAMES or name_lower.startswith("dockerfile"):
        category = "docker"
    else:
        category = _EXT_CATEGORY.get(ext, "other")

    return NonPythonFile(
        path=str(file),
        ext=ext or file.name,
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
# Graph Serialisation
# ===========================================================================

def _write_graph(graph: nx.DiGraph, output_path: Path) -> None:
    """Serialise the NetworkX graph to JSON node-link format."""
    data = nx.node_link_data(graph)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
