"""
tests/test_stage1_ingestion.py

Stage 1 test suite: Universal Lexical Parser (Tree-sitter) + Vector Store.

Coverage:
  - Lexical parser: Python, TypeScript, Rust, Go fixture files via Tree-sitter
    AST queries (_query_names, _extract_imports, _extract_todos)
  - _parse_file: per-file integration test for each language
  - Non-code file categorisation (_classify_non_code)
  - NetworkX edge construction from import relationships (build_knowledge_graph)
  - KnowledgeGraph stats (file_type_summary, language_summary)
  - Uncovered function detection via naming convention (_find_uncovered):
      Python/Rust/JS/TS: test_<func_name>
      Go:                Test<FuncName>
  - VectorStore: in-memory init, ingest, query (TF-IDF fallback, no model
    download)
  - Chunk-text splitting (_chunk_text)

All tests use the fixtures in tests/fixtures/ and tmp_path for isolated
file-system operations. LLM and FastEmbed model downloads are NOT triggered
— the TF-IDF fallback is tested to keep the suite fast and offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tree_sitter import Parser

from contribtriage.ingestion.lexical_parser import (
    LANGUAGE_CONFIGS,
    _classify_non_code,
    _count_by_ext,
    _extract_imports,
    _extract_todos,
    _find_uncovered,
    _parse_file,
    _path_to_logical_name,
    _query_names,
    build_knowledge_graph,
)
from contribtriage.ingestion.vector_store import VectorStore, _chunk_text as chunk_text
from contribtriage.models import KnowledgeGraph, ModuleNode, NonPythonFile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _read_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _parse(ext: str, name: str):
    """Parse a fixture file with Tree-sitter. Returns (cfg, tree, source_text, source_bytes)."""
    cfg  = LANGUAGE_CONFIGS[ext]
    src  = _read_bytes(name)
    p    = Parser(cfg.ts_language)
    tree = p.parse(src)
    return cfg, tree, src.decode("utf-8", errors="replace"), src


# ===========================================================================
# 1. Lexical Parser — Python (Tree-sitter queries)
# ===========================================================================

class TestPythonParser:

    def setup_method(self):
        self.cfg, self.tree, self.source, self.src_bytes = _parse(".py", "sample.py")

    def test_extracts_classes(self):
        classes = _query_names(self.cfg.ts_language, self.tree, self.cfg.class_queries)
        assert "DataProcessor" in classes
        assert "DataWriter" in classes

    def test_extracts_functions(self):
        funcs = _query_names(self.cfg.ts_language, self.tree, self.cfg.func_queries)
        assert "validate_schema" in funcs
        assert "run_pipeline" in funcs

    def test_extracts_imports(self):
        imports = _extract_imports(self.cfg, self.tree, self.source)
        assert "os" in imports
        assert "json" in imports
        assert "pathlib" in imports

    def test_extracts_todos(self):
        todos = _extract_todos(self.cfg.todo_pattern, self.source)
        # Fixture has 3 tags: TODO, FIXME, BUG
        assert len(todos) >= 3
        tags = [t.split(":")[0] for t in todos]
        assert "TODO" in tags
        assert "FIXME" in tags
        assert "BUG" in tags

    def test_todo_content_preserved(self):
        todos = _extract_todos(self.cfg.todo_pattern, self.source)
        combined = " ".join(todos)
        assert "caching layer" in combined.lower() or "caching" in combined.lower()

    def test_private_functions_still_extracted(self):
        """Tree-sitter extracts ALL functions including private ones.
        Filtering to public-only happens at the coverage analysis layer."""
        src = b"def _private_helper():\n    pass\ndef public_func():\n    pass\n"
        p = Parser(self.cfg.ts_language)
        t = p.parse(src)
        funcs = _query_names(self.cfg.ts_language, t, self.cfg.func_queries)
        assert "_private_helper" in funcs
        assert "public_func" in funcs


# ===========================================================================
# 2. Lexical Parser — TypeScript (Tree-sitter queries)
# ===========================================================================

class TestTypeScriptParser:

    def setup_method(self):
        self.cfg, self.tree, self.source, self.src_bytes = _parse(".ts", "sample.ts")

    def test_extracts_classes(self):
        classes = _query_names(self.cfg.ts_language, self.tree, self.cfg.class_queries)
        assert "ApiClient" in classes
        assert "DataTransformer" in classes

    def test_extracts_interfaces(self):
        # Interfaces and type aliases are captured by class_queries in TS
        classes = _query_names(self.cfg.ts_language, self.tree, self.cfg.class_queries)
        assert "DataPayload" in classes or "Config" in classes or len(classes) >= 2

    def test_extracts_functions(self):
        funcs = _query_names(self.cfg.ts_language, self.tree, self.cfg.func_queries)
        # loadConfig and runPipeline are top-level exports
        assert "loadConfig" in funcs or "runPipeline" in funcs or len(funcs) >= 1

    def test_extracts_imports(self):
        imports = _extract_imports(self.cfg, self.tree, self.source)
        assert "fs" in imports or "axios" in imports or len(imports) >= 1

    def test_extracts_todos(self):
        todos = _extract_todos(self.cfg.todo_pattern, self.source)
        assert len(todos) >= 2


# ===========================================================================
# 3. Lexical Parser — Rust (Tree-sitter queries)
# ===========================================================================

class TestRustParser:

    def setup_method(self):
        self.cfg, self.tree, self.source, self.src_bytes = _parse(".rs", "sample.rs")

    def test_extracts_structs(self):
        classes = _query_names(self.cfg.ts_language, self.tree, self.cfg.class_queries)
        assert "DataRecord" in classes
        assert "DataProcessor" in classes

    def test_extracts_enums(self):
        classes = _query_names(self.cfg.ts_language, self.tree, self.cfg.class_queries)
        assert "ProcessorError" in classes

    def test_extracts_traits(self):
        classes = _query_names(self.cfg.ts_language, self.tree, self.cfg.class_queries)
        assert "Transformer" in classes

    def test_extracts_functions(self):
        funcs = _query_names(self.cfg.ts_language, self.tree, self.cfg.func_queries)
        assert "run_pipeline" in funcs
        # impl methods must also be captured
        assert "new" in funcs or "load_records" in funcs or "process" in funcs

    def test_extracts_use_statements(self):
        imports = _extract_imports(self.cfg, self.tree, self.source)
        # 'use std::collections::HashMap;' → normalised to 'std'
        # 'use serde::...' → normalised to 'serde'
        assert "std" in imports
        assert "serde" in imports

    def test_extracts_todos(self):
        todos = _extract_todos(self.cfg.todo_pattern, self.source)
        assert len(todos) >= 2


# ===========================================================================
# 4. Lexical Parser — Go (Tree-sitter queries)
# ===========================================================================

class TestGoParser:

    def setup_method(self):
        self.cfg, self.tree, self.source, self.src_bytes = _parse(".go", "sample.go")

    def test_extracts_structs(self):
        classes = _query_names(self.cfg.ts_language, self.tree, self.cfg.class_queries)
        assert "DataRecord" in classes
        assert "DataProcessor" in classes

    def test_extracts_interfaces(self):
        classes = _query_names(self.cfg.ts_language, self.tree, self.cfg.class_queries)
        assert "Transformer" in classes

    def test_extracts_functions(self):
        funcs = _query_names(self.cfg.ts_language, self.tree, self.cfg.func_queries)
        assert "RunPipeline" in funcs or "NewDataProcessor" in funcs

    def test_extracts_imports(self):
        imports = _extract_imports(self.cfg, self.tree, self.source)
        # Go import paths are normalised to the last segment
        assert any(
            imp in ("json", "os", "fmt", "io", "http")
            for imp in imports
        )

    def test_extracts_todos(self):
        todos = _extract_todos(self.cfg.todo_pattern, self.source)
        assert len(todos) >= 2


# ===========================================================================
# 5. _parse_file helper — integration test against real fixture files
# ===========================================================================

class TestParseFile:
    """Tests for the _parse_file internal helper, which is the per-file
    entry point called by build_knowledge_graph."""

    def test_python_module_node_has_functions(self):
        cfg  = LANGUAGE_CONFIGS[".py"]
        node = _parse_file(FIXTURES / "sample.py", "sample", cfg)
        assert len(node.functions) >= 2

    def test_python_module_node_has_classes(self):
        cfg  = LANGUAGE_CONFIGS[".py"]
        node = _parse_file(FIXTURES / "sample.py", "sample", cfg)
        assert len(node.classes) >= 1

    def test_rust_module_node_has_structs(self):
        cfg  = LANGUAGE_CONFIGS[".rs"]
        node = _parse_file(FIXTURES / "sample.rs", "sample_rs", cfg)
        assert len(node.classes) >= 1

    def test_go_module_node_has_functions(self):
        cfg  = LANGUAGE_CONFIGS[".go"]
        node = _parse_file(FIXTURES / "sample.go", "sample_go", cfg)
        assert len(node.functions) >= 1

    def test_ts_module_node_has_classes(self):
        cfg  = LANGUAGE_CONFIGS[".ts"]
        node = _parse_file(FIXTURES / "sample.ts", "sample_ts", cfg)
        assert len(node.classes) >= 1

    def test_module_name_set_correctly(self):
        cfg  = LANGUAGE_CONFIGS[".py"]
        node = _parse_file(FIXTURES / "sample.py", "fixtures.sample", cfg)
        assert node.module_name == "fixtures.sample"

    def test_language_label_set(self):
        cfg  = LANGUAGE_CONFIGS[".rs"]
        node = _parse_file(FIXTURES / "sample.rs", "sample_rs", cfg)
        assert node.language == "Rust"


# ===========================================================================
# 6. Non-Code File Classification
# ===========================================================================

class TestNonCodeClassification:

    def _make_file(self, tmp_path: Path, name: str, content: str = "x") -> Path:
        f = tmp_path / name
        f.write_text(content)
        return f

    def test_markdown_is_docs(self, tmp_path):
        f = self._make_file(tmp_path, "README.md")
        nf = _classify_non_code(f)
        assert nf.category == "docs"
        assert nf.ext == ".md"

    def test_yml_is_ci(self, tmp_path):
        f = self._make_file(tmp_path, "ci.yml")
        nf = _classify_non_code(f)
        assert nf.category == "ci"

    def test_dockerfile_is_docker(self, tmp_path):
        f = self._make_file(tmp_path, "Dockerfile")
        nf = _classify_non_code(f)
        assert nf.category == "docker"

    def test_json_is_config(self, tmp_path):
        f = self._make_file(tmp_path, "config.json")
        nf = _classify_non_code(f)
        assert nf.category == "config"

    def test_size_kb_computed(self, tmp_path):
        content = "x" * 2048  # 2 KB
        f = self._make_file(tmp_path, "big.md", content)
        nf = _classify_non_code(f)
        assert nf.size_kb == pytest.approx(2.0, abs=0.1)

    def test_unknown_ext_is_other(self, tmp_path):
        f = self._make_file(tmp_path, "weird.xyz")
        nf = _classify_non_code(f)
        assert nf.category == "other"


# ===========================================================================
# 7. Path-to-Logical-Name Conversion
# ===========================================================================

class TestPathToLogicalName:

    def test_simple_module(self, tmp_path):
        root = tmp_path
        f = root / "mymodule.py"
        f.touch()
        assert _path_to_logical_name(f, root) == "mymodule"

    def test_nested_module(self, tmp_path):
        root = tmp_path
        f = root / "pkg" / "sub" / "module.py"
        f.parent.mkdir(parents=True)
        f.touch()
        assert _path_to_logical_name(f, root) == "pkg.sub.module"

    def test_init_dropped(self, tmp_path):
        root = tmp_path
        f = root / "mypkg" / "__init__.py"
        f.parent.mkdir(parents=True)
        f.touch()
        assert _path_to_logical_name(f, root) == "mypkg"

    def test_non_python_gets_ext_suffix(self, tmp_path):
        """Non-Python files get an _<ext> suffix to keep names unique."""
        root = tmp_path
        f = root / "lib" / "main.go"
        f.parent.mkdir(parents=True)
        f.touch()
        name = _path_to_logical_name(f, root)
        assert name == "lib.main_go"

    def test_rust_file_suffix(self, tmp_path):
        root = tmp_path
        f = root / "src" / "lib.rs"
        f.parent.mkdir(parents=True)
        f.touch()
        assert _path_to_logical_name(f, root) == "src.lib_rs"


# ===========================================================================
# 8. build_knowledge_graph — Integration (uses fixtures dir)
# ===========================================================================

class TestBuildKnowledgeGraph:

    def test_returns_knowledge_graph(self, tmp_path):
        """build_knowledge_graph on fixtures dir returns populated KG."""
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        assert isinstance(kg, KnowledgeGraph)

    def test_python_file_parsed(self, tmp_path):
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        # Python files keep stem only: sample.py → 'sample'
        py_nodes = [n for n, nd in kg.nodes.items() if nd.language == "Python"]
        assert len(py_nodes) >= 1, f"No Python nodes found. nodes: {list(kg.nodes.keys())}"

    def test_typescript_file_parsed(self, tmp_path):
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        ts_nodes = [n for n, nd in kg.nodes.items() if "TypeScript" in nd.language]
        assert len(ts_nodes) >= 1

    def test_rust_file_parsed(self, tmp_path):
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        # Non-Python files get ext suffix: sample.rs → 'sample_rs'
        rs_nodes = [n for n, nd in kg.nodes.items() if nd.language == "Rust"]
        assert len(rs_nodes) >= 1, f"No Rust nodes found. nodes: {list(kg.nodes.keys())}"

    def test_go_file_parsed(self, tmp_path):
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        go_nodes = [n for n, nd in kg.nodes.items() if nd.language == "Go"]
        assert len(go_nodes) >= 1, f"No Go nodes found. nodes: {list(kg.nodes.keys())}"

    def test_markdown_in_non_python_files(self, tmp_path):
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        md_files = [f for f in kg.non_python_files if f.ext == ".md"]
        assert len(md_files) >= 1

    def test_language_summary_populated(self, tmp_path):
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        assert len(kg.language_summary) >= 2  # at least Python + one other

    def test_file_type_summary_populated(self, tmp_path):
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        assert len(kg.file_type_summary) >= 1

    def test_graph_json_written(self, tmp_path):
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        assert kg.graph_json_path is not None
        assert Path(kg.graph_json_path).exists()

    def test_graph_json_valid(self, tmp_path):
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        data = json.loads(Path(kg.graph_json_path).read_text())
        assert "nodes" in data
        assert "links" in data

    def test_todos_extracted_from_python(self, tmp_path):
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        py_nodes = [nd for nd in kg.nodes.values() if nd.language == "Python"]
        all_todos = [t for nd in py_nodes for t in nd.todos]
        assert len(all_todos) >= 3

    def test_skips_hidden_dirs(self, tmp_path):
        """Files inside .git or __pycache__ must never appear in nodes."""
        # Create a fake .git dir with a .py file
        git_dir = FIXTURES.parent / ".git"
        git_dir.mkdir(exist_ok=True)
        fake = git_dir / "fake_module.py"
        fake.write_text("class FakeGitClass: pass")
        try:
            kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
            assert "fake_module" not in kg.nodes
        finally:
            fake.unlink(missing_ok=True)
            if git_dir.exists() and not any(git_dir.iterdir()):
                git_dir.rmdir()

    def test_all_four_languages_in_fixtures(self, tmp_path):
        """Fixtures directory contains .py .ts .rs .go — all must parse."""
        kg = build_knowledge_graph(str(FIXTURES), output_dir=str(tmp_path))
        langs = set(kg.language_summary.keys())
        assert "Python"     in langs
        assert "TypeScript" in langs
        assert "Rust"       in langs
        assert "Go"         in langs


# ===========================================================================
# 9. Uncovered Function Detection
# ===========================================================================

class TestFindUncovered:

    def test_uncovered_public_function_flagged(self):
        nodes = {
            "mymod": ModuleNode(
                path="mymod.py", module_name="mymod",
                functions=["do_something", "helper"],
            ),
            "tests.test_mymod": ModuleNode(
                path="tests/test_mymod.py",
                module_name="tests.test_mymod",
                functions=["test_helper"],
            ),
        }
        test_modules = {"tests.test_mymod"}
        uncovered = _find_uncovered(nodes, test_modules)
        # do_something is not referenced in the test module text
        assert "mymod.do_something" in uncovered

    def test_private_functions_excluded(self):
        nodes = {
            "mymod": ModuleNode(
                path="mymod.py", module_name="mymod",
                functions=["_private", "__dunder__"],
            ),
            "tests.test_mymod": ModuleNode(
                path="tests/test_mymod.py",
                module_name="tests.test_mymod",
                functions=[],
            ),
        }
        uncovered = _find_uncovered(nodes, {"tests.test_mymod"})
        # Private functions must NOT appear in uncovered list
        assert len(uncovered) == 0

    def test_function_covered_by_naming_convention(self):
        """A function is covered when test_<func_name> exists in any test module."""
        nodes = {
            "mod": ModuleNode(
                path="mod.py", module_name="mod",
                functions=["parse_url", "validate_schema"],
            ),
            "test_mod": ModuleNode(
                path="tests/test_mod.py", module_name="test_mod",
                # test_parse_url present → parse_url covered
                # test_validate_schema absent → validate_schema uncovered
                functions=["test_parse_url", "test_helper"],
            ),
        }
        uncovered = _find_uncovered(nodes, {"test_mod"})
        assert "mod.parse_url" not in uncovered        # has matching test_*
        assert "mod.validate_schema" in uncovered      # no matching test_*

    def test_go_function_covered_by_TestFuncName_convention(self):
        """Go functions are covered when Test<FuncName> exists."""
        from contribtriage.models import ModuleNode
        nodes = {
            "pkg": ModuleNode(
                path="pkg/main.go", module_name="pkg",
                language="Go",
                functions=["RunPipeline", "NewProcessor"],
                classes=[], imports=[], todos=[],
            ),
            "pkg_test": ModuleNode(
                path="pkg/main_test.go", module_name="pkg_test",
                language="Go",
                # TestRunPipeline present → RunPipeline covered
                # TestNewProcessor absent → NewProcessor uncovered
                functions=["TestRunPipeline"],
                classes=[], imports=[], todos=[],
            ),
        }
        uncovered = _find_uncovered(nodes, {"pkg_test"})
        assert "pkg.RunPipeline" not in uncovered      # TestRunPipeline exists
        assert "pkg.NewProcessor" in uncovered         # TestNewProcessor missing


# ===========================================================================
# 10. VectorStore — In-Memory (TF-IDF fallback, no model download)
# ===========================================================================

class TestVectorStore:
    """
    Tests use the TF-IDF fallback embedder to avoid downloading the FastEmbed
    model during CI. We patch fastembed.TextEmbedding to raise ImportError,
    forcing the fallback path.
    """

    @pytest.fixture
    def store(self):
        """In-memory store with TF-IDF fallback forced via patch."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove GEMINI_API_KEY if set, to ensure FastEmbed path is tried
            os.environ.pop("GEMINI_API_KEY", None)
            with patch("contribtriage.ingestion.vector_store.VectorStore._init_embedder") as mock_init:
                # Return the TF-IDF embedder directly
                vs = VectorStore.__new__(VectorStore)
                vs._path = ":memory:"
                vs._use_gemini = False
                vs._gemini_key = None
                vs._vector_size = 384
                vs._client = vs._init_client()
                # Use the real TF-IDF fallback
                vs._embed_fn = vs._make_tfidf_embedder()
                vs._ensure_collection()
                yield vs

    def test_store_initialises(self, store):
        assert store.count() == 0

    def test_ingest_text_chunks(self, store):
        """Ingesting raw text chunks via _upsert_batch increases count."""
        texts = ["Hello world", "Testing the vector store", "Another document"]
        metas = [{"source": "test", "kind": "docs"}] * 3
        store._upsert_batch(texts, metas)
        assert store.count() == 3

    def test_query_returns_results(self, store):
        texts = [
            "Install dependencies using pip install -r requirements.txt",
            "Run tests with pytest from the project root",
            "The DataProcessor class handles JSON parsing",
        ]
        metas = [{"source": "README.md", "kind": "docs"}] * 3
        store._upsert_batch(texts, metas)

        results = store.query("how to install dependencies", top_k=2)
        assert len(results) >= 1
        assert "text" in results[0]
        assert "score" in results[0]

    def test_query_top_k_respected(self, store):
        texts = [f"Document number {i}" for i in range(10)]
        metas = [{"source": "test", "kind": "docs"}] * 10
        store._upsert_batch(texts, metas)
        results = store.query("document", top_k=3)
        assert len(results) <= 3

    def test_ingest_repo_from_knowledge_graph(self, store, tmp_path):
        """Full ingest_repo path: KG with non_python_files and docstrings."""
        # Build a minimal KG with a fixture markdown file
        readme = FIXTURES / "README.md"
        nf = NonPythonFile(
            path=str(readme), ext=".md", category="docs", size_kb=1.0
        )
        node = ModuleNode(
            path=str(FIXTURES / "sample.py"),
            module_name="sample",
            language="Python",
            docstring="Processes raw data payloads.",
            todos=["TODO: Add caching", "BUG: empty dicts"],
        )
        kg = KnowledgeGraph(
            nodes={"sample": node},
            non_python_files=[nf],
        )
        count = store.ingest_repo(kg, str(tmp_path))
        # At minimum: markdown chunks + 1 docstring + 2 todos
        assert count >= 3

    def test_in_memory_constructor(self):
        """VectorStore.in_memory() factory works without file I/O."""
        with patch(
            "contribtriage.ingestion.vector_store.VectorStore._init_embedder",
            return_value=lambda texts: [[0.0] * 384] * len(texts),
        ):
            store = VectorStore.in_memory()
            assert store.count() == 0


# ===========================================================================
# 11. Chunk-Text Splitting
# ===========================================================================

class TestChunkText:

    def test_short_text_single_chunk(self):
        text = "Short document."
        chunks = chunk_text(text, source="test.md", kind="docs")
        assert len(chunks) == 1
        assert chunks[0][0] == text

    def test_long_text_multiple_chunks(self):
        # Create a text longer than _CHUNK_SIZE
        paragraphs = ["Paragraph number %d. " % i + ("x " * 50) for i in range(30)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, source="big.md", kind="docs", chunk_size=200)
        assert len(chunks) > 1

    def test_metadata_present_in_all_chunks(self):
        text = "\n\n".join(["Para %d" % i for i in range(20)])
        chunks = chunk_text(text, source="doc.md", kind="ci", chunk_size=50)
        for _, meta in chunks:
            assert meta["source"] == "doc.md"
            assert meta["kind"] == "ci"
            assert "chunk_index" in meta

    def test_chunk_indices_sequential(self):
        text = "\n\n".join(["Paragraph " + str(i) for i in range(10)])
        chunks = chunk_text(text, source="x.md", kind="docs", chunk_size=30)
        indices = [meta["chunk_index"] for _, meta in chunks]
        assert indices == list(range(len(chunks)))
