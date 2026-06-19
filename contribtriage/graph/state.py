"""
contribtriage/graph/state.py

Re-exports LangGraphState from models.py.

The canonical definition lives in models.py (alongside all other dataclasses)
so that every module can import it without a circular dependency on graph/.
This module exists as the canonical reference point for graph-specific code.
"""

from contribtriage.models import LangGraphState

__all__ = ["LangGraphState"]
