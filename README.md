# ContribTriage

**ContribTriage** is a CLI tool that points at any cloned repository and produces a **`SETUP_DIAGNOSTICS.md`** report that helps new contributors get their environment running fast. It is powered by a **LangGraph StateGraph** pipeline that combines Tree-sitter AST parsing, a Qdrant vector store, Groq Llama-3 real-time analysis, and Gemini 2.5 Flash report synthesis — automating everything from environment auditing to AI-generated onboarding guides.

## 🔹 Key Features

- **Polyglot AST Parsing** - Tree-sitter grammars for Python, JS, TS, Rust, and Go parse the full codebase without regex fragility.
- **Knowledge Graph** - NetworkX module-level import graph tracks functions, dependencies, and coverage gaps.
- **Vector Store** - FastEmbed (CPU-only) + Qdrant for semantic retrieval of code context.
- **AI Test Generation** - Groq Llama-3.3-70b generates import-health checks and stub test files per module.
- **Self-Healing Loop** - Detects `ModuleNotFoundError`, proposes fix commands, reruns — up to N retries.
- **Failure Triage** - Groq classifies failures as `code_bug` / `app_dep` / `system_dep` with fix recommendations.
- **Smart Report** - Gemini 2.5 Flash synthesises the full `SETUP_DIAGNOSTICS.md` onboarding report.
- **LangGraph Engine** - All stages run as a compiled StateGraph — resumable via `--persist` checkpointing.

## 🔹 System Architecture

### 1. Codebase Ingestion → Knowledge Graph + Vector Store
Tree-sitter parses every source file into an AST. Functions, imports, and modules are indexed into a NetworkX knowledge graph. Code chunks are embedded via FastEmbed and stored in Qdrant for semantic search.

### 2. Manifest + Environment Audit
Project manifests (`pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`) are parsed for declared dependencies. The host environment (OS, runtimes, venv, Docker, system tools) is snapshotted.

### 3. AI Test Generation + Test Runner
Groq generates one test file per module (import-health + stubs for uncovered public functions). The test suite is executed via `pytest` / `cargo test` / `go test` / `npm test`.

### 4. Heal-and-Rerun Loop
On failure, Groq reads the full `terminal_log_history` — every previous run's stdout/stderr — classifies the failure, and proposes a fix. The user consents, the fix is applied, and the suite reruns. The loop exits when:
- All tests pass → route to report
- Failure classified as `code_bug` → route to report (surfaces as contribution opportunity)
- User declines fix (`N` at prompt) → route to report
- `retry_count >= max_retries` → route to report

### 5. End-to-End Flow
Developer clones repo → ContribTriage ingests → audits environment → generates & runs tests → heals dependencies → Gemini synthesises `SETUP_DIAGNOSTICS.md`.

### Example session

```
╭──────────────────────────────── 🚀  ContribTriage ───────────────────────────╮
│ ContribTriage v2                                                              │
│ Repo: /home/user/projects/httpie                                              │
│ Max retries: 3  Auto-accept: no  Persist: no                                 │
│ LangGraph StateGraph engine starting…                                         │
╰───────────────────────────────────────────────────────────────────────────────╯
  [Stage 1] Parsing codebase: /home/user/projects/httpie
  -> 42 code files, 18 other files found
  ✓ Vector store: 183 chunks ingested
  ✓ KG: 42 module(s) indexed
  [Stage 2] Parsing project manifests…
  ✓ 12 declared dep(s) found
  [Stage 3] Auditing host environment…
  ✓ Linux | Python 3.11.4 | Docker ✓
  ✓ Generated 5 test file(s) (imports + stubs, AI-powered)
  [Stage 4/5] Full test suite [pytest]…
  ✓ Tests: 38 passed / 0 failed / 0 errors (38 total)
  [Stage 6] Generating SETUP_DIAGNOSTICS.md…
  ✓ SETUP_DIAGNOSTICS.md synthesised by Gemini

✓  Done!  Report: /home/user/projects/httpie/SETUP_DIAGNOSTICS.md
```

---

## 🔹 Tech Stack

**CLI & Pipeline (Python):**
- LangGraph StateGraph (pipeline engine + checkpointing)
- Tree-sitter (polyglot AST parsing)
- NetworkX (knowledge graph)
- FastEmbed + Qdrant (vector store, CPU-only)

**AI & Analysis:**
- Groq Llama-3.3-70b (test generation, failure triage, fix classification)
- Gemini 2.5 Flash (report synthesis)

**Test Runners:**
- pytest / unittest (Python)
- cargo test (Rust)
- go test (Go)
- npm test / Jest (JS/TS)

**Supported Languages:**
- Python, JavaScript, TypeScript, Rust, Go

## 🔹 Directory Structure

```
ContribTriage/
├── contribtriage/
│   ├── cli.py                  # CLI entry point — arg parsing → build_graph().invoke()
│   ├── models.py               # All dataclasses + LangGraphState TypedDict
│   ├── exceptions.py           # Custom exception hierarchy
│   ├── graph/
│   │   ├── __init__.py         # build_graph() — compiles the StateGraph
│   │   ├── nodes.py            # All 9 LangGraph node functions (Stages 1–6)
│   │   ├── edges.py            # Conditional routing functions (decision logic)
│   │   └── state.py            # State schema alias
│   ├── ingestion/
│   │   ├── lexical_parser.py   # Tree-sitter AST parser → KnowledgeGraph
│   │   ├── vector_store.py     # Qdrant + FastEmbed vector store wrapper
│   │   └── doc_reader.py       # Manifest parser (pyproject.toml, package.json, etc.)
│   ├── runners/
│   │   ├── test_runner.py      # Subprocess test runner + output parser
│   │   └── test_generator.py   # Groq-powered AI test stub generator
│   ├── agents/
│   │   ├── groq_agent.py       # Groq Llama-3 failure analysis + fix classification
│   │   └── gemini_agent.py     # Gemini 2.5 Flash report synthesis
│   ├── audit/
│   │   └── env_auditor.py      # Host environment snapshot (OS, runtimes, Docker, tools)
│   ├── resolver/
│   │   └── installer.py        # Fix command executor (pip/npm/cargo/go install)
│   └── report/
│       └── report_generator.py # SETUP_DIAGNOSTICS.md writer
├── tests/
│   ├── test_stage1_ingestion.py
│   ├── test_stage2_manifest.py
│   ├── test_stage3_audit.py
│   ├── test_stage4_runners.py
│   ├── test_stage5_graph.py
│   └── fixtures/               # Sample repos, manifests, and source files for testing
├── pyproject.toml
└── README.md
```

## 🔹 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/ContribTriage
cd ContribTriage
```

### 2. Install ContribTriage
```bash
pip install -e .

# With dev dependencies (required for running tests)
pip install -e ".[dev]"
```

### 3. Configure Environment Variables
Create a `.env` file at the repo root:
```bash
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...   # Optional — report falls back to a plain template if absent
```

### 4. Run ContribTriage
```bash
# Basic — analyse a cloned repo
contribtriage --repo-path ./path/to/cloned-repo

# Non-interactive mode (auto-accept all fix commands)
contribtriage --repo-path ./httpie --yes

# Increase heal cycles and enable checkpointing
contribtriage --repo-path ./next.js --max-retries 5 --persist
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--repo-path PATH` | *(required)* | Path to the cloned repository |
| `--yes` | `false` | Auto-accept all proposed fix commands (CI-friendly) |
| `--max-retries N` | `3` | Maximum heal-and-rerun cycles before routing to report |
| `--persist` | `false` | Enable SqliteSaver checkpointing for resumable runs |

### 5. Run Tests
```bash
pytest
pytest --cov=contribtriage --cov-report=term-missing
```
