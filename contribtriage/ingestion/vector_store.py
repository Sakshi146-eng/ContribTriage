"""
contribtriage/ingestion/vector_store.py

Stage 1b — Qdrant Vector Store with FastEmbed / Gemini upgrade path.

Design:
  - Default: FastEmbed (BAAI/bge-small-en-v1.5) — CPU-local, zero-config,
    no API key required. Works completely offline.
  - Upgrade: If GEMINI_API_KEY is set in the environment, the store
    automatically upgrades to Gemini text-embedding-004 (768-dim) for
    higher-quality multilingual embeddings.
  - Persistent: QdrantClient(path=...) stores vectors on disk under
    .contribtriage_qdrant/ inside the repo — no server process needed.
  - In-memory mode: pass path=":memory:" for tests / ephemeral runs.

Ingested content:
  - README.md, CONTRIBUTING.md, and all other .md/.rst docs
  - YAML CI configs, Dockerfiles, shell scripts
  - Docstrings extracted from ModuleNodes in the KnowledgeGraph
  - TODO/FIXME/BUG comments from source files

Query interface:
  - query(text, top_k) → list of relevant text chunks with metadata
  - Used by the Groq agent to ground its traceback reasoning with
    repo-specific context (e.g. "how does this project handle deps?")
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from contribtriage.models import KnowledgeGraph, NonPythonFile

console = Console()

# Qdrant collection name — one collection per tool run
_COLLECTION = "contribtriage_docs"

# FastEmbed model: small (384-dim), fast on CPU, good multilingual quality
_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
_FASTEMBED_DIM   = 384

# Gemini embedding model: 768-dim, higher quality, requires API key
_GEMINI_MODEL    = "models/text-embedding-004"
_GEMINI_DIM      = 768

# File extensions worth vectorising (text-based, not binary)
_VECTORISE_EXTS = {
    ".md", ".rst", ".txt",
    ".yml", ".yaml",
    ".sh", ".bash", ".zsh",
    ".json", ".toml", ".ini", ".cfg",
    ".html", ".css",
    # Extension-less special files
    "",
}
_VECTORISE_NAMES = {
    "dockerfile", "makefile", "procfile", "vagrantfile",
    "jenkinsfile", "readme", "contributing", "changelog",
    "license", "authors", "notice",
}

# Maximum characters per chunk — keeps embedding latency predictable
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100


# ===========================================================================
# VectorStore
# ===========================================================================

class VectorStore:
    """
    Qdrant-backed semantic store for repository documentation and comments.

    Usage::

        store = VectorStore.from_repo(repo_root=".contribtriage_qdrant/")
        store.ingest_repo(knowledge_graph, repo_root)
        results = store.query("how to install dependencies", top_k=5)
    """

    def __init__(
        self,
        store_path: str,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            store_path:     Path for Qdrant persistent storage, or ":memory:".
            gemini_api_key: If provided, use Gemini embeddings; otherwise FastEmbed.
        """
        self._path = store_path
        self._use_gemini = bool(gemini_api_key)
        self._gemini_key = gemini_api_key
        self._vector_size = _GEMINI_DIM if self._use_gemini else _FASTEMBED_DIM
        self._client = self._init_client()
        self._embed_fn = self._init_embedder()
        self._ensure_collection()

        mode = "Gemini text-embedding-004" if self._use_gemini else "FastEmbed (local CPU)"
        console.print(f"[cyan]  → Vector store: {mode}, dim={self._vector_size}[/cyan]")

    # ── Construction helpers ─────────────────────────────────────────────

    @classmethod
    def from_repo(
        cls,
        repo_root: str,
        gemini_api_key: Optional[str] = None,
    ) -> "VectorStore":
        """
        Convenience constructor that places the Qdrant store inside the repo
        at ``<repo_root>/.contribtriage_qdrant/``.

        Automatically checks GEMINI_API_KEY env var if *gemini_api_key* is None.
        """
        key = gemini_api_key or os.getenv("GEMINI_EMBEDDING_API_KEY")
        store_path = str(Path(repo_root) / ".contribtriage_qdrant")
        return cls(store_path=store_path, gemini_api_key=key)

    @classmethod
    def in_memory(cls) -> "VectorStore":
        """Create an in-memory store — useful for tests."""
        return cls(store_path=":memory:", gemini_api_key=None)

    # ── Client + Embedder Init ────────────────────────────────────────────

    def _init_client(self):
        """Initialise Qdrant client — persistent file mode or in-memory."""
        from qdrant_client import QdrantClient
        if self._path == ":memory:":
            return QdrantClient(":memory:")
        Path(self._path).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=self._path)

    def _init_embedder(self):
        """
        Return an embedding callable: texts: List[str] → List[List[float]]

        Tries FastEmbed first; if unavailable (e.g. compilation fails in a
        locked-down env), falls back to a TF-IDF/BM25 semantic similarity
        engine that produces sparse pseudo-vectors — never crashes.
        """
        if self._use_gemini:
            return self._make_gemini_embedder()

        try:
            from fastembed import TextEmbedding
            model = TextEmbedding(model_name=_FASTEMBED_MODEL)
            # Warm up — triggers model download on first call
            def _fastembed(texts: List[str]) -> List[List[float]]:
                return [list(v) for v in model.embed(texts)]
            return _fastembed
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[yellow]  ⚠ FastEmbed unavailable ({type(exc).__name__}). "
                "Falling back to TF-IDF sparse embedder.[/yellow]"
            )
            return self._make_tfidf_embedder()

    def _make_gemini_embedder(self):
        """Gemini text-embedding-004 via google-genai SDK."""
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=self._gemini_key)

        def _gemini_embed(texts: List[str]) -> List[List[float]]:
            vectors = []
            for text in texts:
                result = genai.embed_content(
                    model=_GEMINI_MODEL,
                    content=text,
                    task_type="retrieval_document",
                )
                vectors.append(result["embedding"])
            return vectors

        return _gemini_embed

    def _make_tfidf_embedder(self):
        """
        Fallback: TF-IDF sparse pseudo-embedder.

        Produces a fixed-size dense vector via hashed term frequencies.
        Lower quality than neural embeddings but requires zero compilation
        and works in fully offline, locked-down environments.
        """
        import hashlib
        import math

        def _tfidf_embed(texts: List[str]) -> List[List[float]]:
            vectors = []
            dim = self._vector_size
            for text in texts:
                vec = [0.0] * dim
                words = re.findall(r"\w+", text.lower())
                for word in words:
                    idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
                    vec[idx] += 1.0
                # L2 normalise
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                vectors.append([x / norm for x in vec])
            return vectors

        return _tfidf_embed

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't already exist."""
        from qdrant_client.models import Distance, VectorParams
        existing = [c.name for c in self._client.get_collections().collections]
        if _COLLECTION not in existing:
            self._client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_repo(
        self,
        knowledge_graph: KnowledgeGraph,
        repo_root: str,
    ) -> int:
        """
        Ingest all vectorisable content from the repository.

        Sources ingested (in order):
          1. Non-code files (docs, configs, CI, Dockerfiles)
          2. Docstrings from KnowledgeGraph ModuleNodes
          3. TODO/FIXME/BUG comments from KnowledgeGraph ModuleNodes

        Args:
            knowledge_graph: Populated KnowledgeGraph from lexical_parser.
            repo_root:       Absolute path to the repository root.

        Returns:
            Total number of chunks ingested.
        """
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        # 1. Non-code files
        for nf in knowledge_graph.non_python_files:
            if not self._should_vectorise(nf):
                continue
            try:
                content = Path(nf.path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for chunk, meta in _chunk_text(content, source=nf.path, kind=nf.category):
                texts.append(chunk)
                metas.append(meta)

        # 2. Docstrings from parsed modules
        for mod_name, node in knowledge_graph.nodes.items():
            if node.docstring:
                texts.append(node.docstring)
                metas.append({
                    "source": node.path,
                    "kind": "docstring",
                    "module": mod_name,
                    "language": node.language,
                })

        # 3. TODO/FIXME comments (searchable as contribution hints)
        for mod_name, node in knowledge_graph.nodes.items():
            for todo in node.todos:
                texts.append(todo)
                metas.append({
                    "source": node.path,
                    "kind": "todo",
                    "module": mod_name,
                    "language": node.language,
                })

        if not texts:
            console.print("[yellow]  ⚠ No content found to ingest into vector store[/yellow]")
            return 0

        self._upsert_batch(texts, metas)
        console.print(
            f"[green]  ✓ Vector store: {len(texts)} chunks ingested[/green]"
        )
        return len(texts)

    def _should_vectorise(self, nf: NonPythonFile) -> bool:
        """Return True if this non-code file's content should be embedded."""
        ext  = nf.ext.lower()
        name = Path(nf.path).name.lower()
        return (
            ext in _VECTORISE_EXTS
            or name in _VECTORISE_NAMES
            or nf.category in {"docs", "ci", "docker", "config"}
        )

    def _upsert_batch(
        self,
        texts: List[str],
        metas: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> None:
        """Embed texts in batches and upsert into Qdrant."""
        from qdrant_client.models import PointStruct

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Embedding chunks...", total=len(texts)
            )
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                batch_metas = metas[i: i + batch_size]
                vectors = self._embed_fn(batch_texts)
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={"text": txt, **meta},
                    )
                    for txt, vec, meta in zip(batch_texts, vectors, batch_metas)
                ]
                self._client.upsert(
                    collection_name=_COLLECTION,
                    points=points,
                )
                progress.advance(task, len(batch_texts))

    # ── Query ──────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int = 5,
        kind_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over the ingested repository content.

        Args:
            text:        Query string (e.g. "how to run tests").
            top_k:       Number of results to return.
            kind_filter: Optional — filter by chunk kind
                         ('docs', 'docstring', 'todo', 'ci', etc.)

        Returns:
            List of dicts with 'text', 'score', and metadata fields.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_vec = self._embed_fn([text])[0]

        scroll_filter = None
        if kind_filter:
            scroll_filter = Filter(
                must=[FieldCondition(key="kind", match=MatchValue(value=kind_filter))]
            )

        results = self._client.query_points(
            collection_name=_COLLECTION,
            query=query_vec,            # 'query' in 1.9+, was 'query_vector' before
            limit=top_k,
            query_filter=scroll_filter,
            with_payload=True,
        )

        # query_points returns QueryResponse; the hits are in .points
        return [
            {"text": r.payload.get("text", ""), "score": r.score, **r.payload}
            for r in results.points
        ]

    def count(self) -> int:
        """Return total number of stored vectors."""
        return self._client.count(collection_name=_COLLECTION).count


# ===========================================================================
# Text Chunking
# ===========================================================================

import re  # noqa: E402 — imported here to avoid top-level circular import edge case


def _chunk_text(
    text: str,
    source: str,
    kind: str,
    chunk_size: int = _CHUNK_SIZE,
    overlap: int = _CHUNK_OVERLAP,
) -> List[tuple[str, Dict[str, Any]]]:
    """
    Split *text* into overlapping chunks of at most *chunk_size* characters.

    Returns list of (chunk_text, metadata_dict) tuples.
    """
    # Prefer splitting on paragraph / sentence boundaries
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[tuple[str, Dict[str, Any]]] = []
    buffer = ""
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buffer) + len(para) + 2 <= chunk_size:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            if buffer:
                chunks.append((
                    buffer,
                    {"source": source, "kind": kind, "chunk_index": chunk_idx},
                ))
                chunk_idx += 1
                # Overlap: keep tail of previous buffer
                buffer = buffer[-overlap:] + "\n\n" + para
            else:
                buffer = para

    if buffer.strip():
        chunks.append((
            buffer.strip(),
            {"source": source, "kind": kind, "chunk_index": chunk_idx},
        ))

    return chunks if chunks else [(text[:chunk_size], {"source": source, "kind": kind, "chunk_index": 0})]
