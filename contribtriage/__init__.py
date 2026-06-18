"""ContribTriage — Automated open-source contributor onboarding.

Runs a git-cloned repository through a LangGraph pipeline that:
  1. Maps the codebase via Universal Lexical Parser → NetworkX knowledge graph
  2. Embeds docs/configs into Qdrant vector store (local FastEmbed)
  3. Parses all language manifests (requirements.txt, package.json, Cargo.toml, go.mod)
  4. Audits the local environment (OS, venv, runtimes, system tools)
  5. Runs tests across Python/Node/Rust/Go ecosystems
  6. Heals dependency gaps via uv / npm / cargo / go (with user consent)
  7. Generates SETUP_DIAGNOSTICS.md via Gemini 2.5 Flash
"""

__version__ = "0.1.0"
__author__  = "ContribTriage Contributors"
