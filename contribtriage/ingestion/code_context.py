"""
contribtriage/ingestion/code_context.py

The knowledge graph stores only names + line ranges. These helpers map a
(file, line) to its enclosing function and read the code from disk on demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from contribtriage.models import KnowledgeGraph, ModuleNode

MAX_LINES = 60


def read_lines(path: str, start: int, end: int, max_lines: int = MAX_LINES) -> str:
    """Return lines start..end (1-based, inclusive), numbered, capped at max_lines."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    start = max(start, 1)
    end = min(end, start + max_lines - 1, len(lines))
    return "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1))


def find_function(
    kg: KnowledgeGraph, file_path: str, line: int, repo_root: str
) -> Optional[Tuple[ModuleNode, str, int, int]]:
    """Return (module, name, start, end) of the innermost function containing `line`."""
    p = Path(file_path)
    target = (p if p.is_absolute() else Path(repo_root) / p).resolve()
    for node in kg.nodes.values():
        if Path(node.path).resolve() != target:
            continue
        best = None
        for name, start, end in node.func_spans:
            if start <= line <= end and (best is None or end - start < best[2] - best[1]):
                best = (name, start, end)
        return (node, best[0], best[1], best[2]) if best else None
    return None


def function_source(
    kg: KnowledgeGraph, module_name: str, func_name: str, max_lines: int = MAX_LINES
) -> str:
    """Return the numbered source of `func_name` in `module_name`, or ''."""
    node = kg.nodes.get(module_name)
    if not node:
        return ""
    for name, start, end in node.func_spans:
        if name == func_name:
            return read_lines(node.path, start, end, max_lines)
    return ""