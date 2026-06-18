"""
contribtriage/ingestion/__init__.py

Stage 1 + 2 — Ingestion & Manifest Parsing.

Exposes the three primary entry points:
  - build_knowledge_graph : Universal Lexical Parser → NetworkX KnowledgeGraph
  - VectorStore           : Qdrant persistent client + FastEmbed / Gemini embeddings
  - parse_manifests       : Multi-ecosystem manifest parser → ProjectMeta
"""

from contribtriage.ingestion.lexical_parser import build_knowledge_graph
from contribtriage.ingestion.vector_store import VectorStore
from contribtriage.ingestion.doc_reader import parse_manifests

__all__ = ["build_knowledge_graph", "VectorStore", "parse_manifests"]
