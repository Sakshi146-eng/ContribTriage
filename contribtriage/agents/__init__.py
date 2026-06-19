"""
contribtriage/agents/__init__.py

LLM Agent drivers — Stage 5 (Groq) and Stage 6 (Gemini).
"""

from contribtriage.agents.groq_agent import analyze_failure
from contribtriage.agents.gemini_agent import synthesize_report

__all__ = ["analyze_failure", "synthesize_report"]
