"""
contribtriage/runners/test_generator.py

Stage 4b — AI-Powered Test Stub Generator.

Architecture
────────────
For EACH source module that has uncovered public functions, Groq
(Llama-3.3-70b-versatile) generates ONE test file that contains:

  Part 1 — Import block
    Actual import/require/use statements for the source module AND all its
    third-party dependencies at the TOP of the file.  If any package is
    missing from the environment, the test runner will fail immediately here
    before running a single test — making this a combined import-health check
    AND test scaffold.

  Part 2 — Test stubs
    One minimal test stub per uncovered public function.  Each stub
    either raises NotImplementedError (Python), throws Error (JS), calls
    t.Skip (Go), or panics (Rust).

Syntax validation
─────────────────
  After Groq returns, the generated file is validated before writing:
    Python  → ast.parse()
    JS/TS   → node --check  (requires Node on PATH)
    Go      → gofmt -e      (requires gofmt on PATH)
    Rust    → rustfmt --check (requires rustfmt on PATH)

  If the validator is unavailable, the file is written as-is (the test
  runner will surface any syntax errors naturally).

  If validation FAILS, a fallback template is written to disk instead
  (with a diagnostic comment) so the healing loop can still run against it.

Output locations
────────────────
  Python   →  tests/stubs/test_<module>_stubs.py
  JS/JSX   →  tests/stubs/<module>.test.js
  TS/TSX   →  tests/stubs/<module>.test.ts
  Go       →  tests/stubs/<module>_test.go
  Rust     →  tests/stubs/<module>_test.rs

Design decisions
────────────────
  - One Groq call per source module (not per function) — batches all
    uncovered functions for that file into a single prompt.
  - generate_dep_stubs() is removed — dep testing is now embedded in the
    import block of each module's generated test file.
  - Returns a list of generated file paths (empty list if nothing to stub).
  - Never raises — OS / API errors are logged and skipped per file.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from contribtriage.ingestion.lexical_parser import KnowledgeGraph
from contribtriage.models import ModuleNode, ProjectMeta

console = Console()

_STUB_SUBDIR = "stubs"

# Language label → file extension for generated test files
_TEST_EXTENSION: Dict[str, str] = {
    "Python":               ".py",
    "JavaScript":           ".js",
    "JavaScript/React":     ".js",
    "TypeScript":           ".ts",
    "TypeScript/React":     ".ts",
    "Rust":                 ".rs",
    "Go":                   ".go",
}

# Language label → how to name the test file
_TEST_FILENAME_PATTERN: Dict[str, str] = {
    "Python":               "test_{stem}_stubs.py",
    "JavaScript":           "{stem}.test.js",
    "JavaScript/React":     "{stem}.test.js",
    "TypeScript":           "{stem}.test.ts",
    "TypeScript/React":     "{stem}.test.ts",
    "Rust":                 "{stem}_test.rs",
    "Go":                   "{stem}_test.go",
}


# ===========================================================================
# Public API
# ===========================================================================

def generate_module_test_files(
    knowledge_graph: Optional[KnowledgeGraph],
    project_meta: Optional[ProjectMeta],
    groq_client: Any,
    repo_root: str,
    tests_dir: str = "tests",
) -> List[str]:
    """
    Generate one Groq-powered test file per source module that has uncovered
    public functions.

    Args:
        knowledge_graph: Populated KG from Stage 1. Uses kg.uncovered_funcs
                         and kg.nodes for module details.
        project_meta:    Stage 2 manifest data (test_framework, declared_deps).
        groq_client:     Initialised groq.Groq() client injected from state.
        repo_root:       Absolute path to the repository root.
        tests_dir:       Test directory name (default 'tests').

    Returns:
        List of absolute paths to generated stub files (may be empty).
    """
    if not knowledge_graph:
        console.print("[dim]  → No knowledge graph — skipping stub generation.[/dim]")
        return []

    uncovered = getattr(knowledge_graph, "uncovered_funcs", [])
    if not uncovered:
        console.print("[dim]  → No uncovered functions — skipping stub generation.[/dim]")
        return []

    # Group uncovered functions by source module name
    grouped: Dict[str, List[str]] = defaultdict(list)
    for qualified in uncovered:
        module_name = qualified.rsplit(".", 1)[0] if "." in qualified else "unknown"
        func_name   = qualified.rsplit(".", 1)[-1]
        if not func_name.startswith("_"):   # skip private
            grouped[module_name].append(func_name)

    root     = Path(repo_root).resolve()
    stub_dir = root / tests_dir / _STUB_SUBDIR
    stub_dir.mkdir(parents=True, exist_ok=True)

    generated: List[str] = []

    for module_name, func_names in grouped.items():
        module_node = knowledge_graph.nodes.get(module_name)
        if module_node is None:
            continue

        path = _generate_one_module_file(
            module_node=module_node,
            func_names=func_names,
            project_meta=project_meta,
            groq_client=groq_client,
            stub_dir=stub_dir,
        )
        if path:
            generated.append(path)

    console.print(
        f"[green]  ✓ Generated {len(generated)} test file(s) in "
        f"{stub_dir.relative_to(root)}[/green]"
    )
    return generated


# ===========================================================================
# Per-Module Generator
# ===========================================================================

def _generate_one_module_file(
    module_node: ModuleNode,
    func_names: List[str],
    project_meta: Optional[ProjectMeta],
    groq_client: Any,
    stub_dir: Path,
) -> Optional[str]:
    """
    Generate and write one test file for *module_node*.

    Returns the absolute file path on success, None on failure.
    """
    language    = module_node.language
    module_stem = Path(module_node.path).stem.replace(".test", "").replace("_test", "")
    filename    = _TEST_FILENAME_PATTERN.get(language, "test_{stem}_stubs.py").format(stem=module_stem)
    file_path   = stub_dir / filename

    # Idempotent — skip if file already exists
    if file_path.exists():
        console.print(f"[dim]  → {file_path.name} already exists — skipping.[/dim]")
        return str(file_path)

    # Build Groq prompt
    prompt = _build_prompt(module_node, func_names, project_meta)

    # Call Groq
    content = _call_groq(groq_client, prompt, module_node)

    if content is None:
        content = _make_fallback(module_node, func_names, reason="Groq API unavailable")

    # Strip accidental markdown fences
    content = _strip_fences(content)

    # Validate syntax
    valid, error_msg = _validate_syntax(content, language)
    if not valid:
        console.print(
            f"[yellow]  ⚠ Syntax error in AI output for {file_path.name}: "
            f"{error_msg} — writing fallback[/yellow]"
        )
        content = _make_fallback(module_node, func_names, reason=f"AI syntax error: {error_msg}")

    # Write to disk
    try:
        file_path.write_text(content, encoding="utf-8")
        console.print(f"[blue]  + {file_path.name}[/blue] ({len(func_names)} stub(s))")
        return str(file_path)
    except OSError as exc:
        console.print(f"[yellow]  ⚠ Could not write {file_path.name}: {exc}[/yellow]")
        return None


# ===========================================================================
# Groq Prompt Builder
# ===========================================================================

def _build_prompt(
    module_node: ModuleNode,
    func_names: List[str],
    project_meta: Optional[ProjectMeta],
) -> str:
    """Build the system+user prompt sent to Groq for stub generation."""
    language       = module_node.language
    file_path      = module_node.path
    module_imports = module_node.imports[:20]   # cap to avoid token overrun
    declared_deps  = getattr(project_meta, "declared_deps", [])[:20] if project_meta else []
    framework_list = getattr(project_meta, "test_framework", []) if project_meta else []
    valid_fw       = [f for f in framework_list if f and f != "(none detected)"]
    framework      = valid_fw[0] if valid_fw else _default_framework(language)

    func_list_str  = "\n".join(f"  - {f}" for f in func_names)
    import_str     = ", ".join(module_imports) if module_imports else "(none detected)"
    dep_str        = ", ".join(declared_deps)  if declared_deps  else "(none declared)"

    format_rules   = _format_rules(language, framework)

    return (
        f"You are an expert {language} developer. "
        f"Generate a complete, syntactically valid test file.\n\n"
        f"Source module: {file_path}\n"
        f"Language: {language}\n"
        f"Test framework: {framework}\n\n"
        f"Uncovered public functions to test:\n{func_list_str}\n\n"
        f"Imports used by the source module: {import_str}\n"
        f"Project declared dependencies: {dep_str}\n\n"
        f"STRICT REQUIREMENTS:\n"
        f"1. At the VERY TOP of the file, write import/require/use statements for:\n"
        f"   - The source module itself\n"
        f"   - All third-party dependencies it uses\n"
        f"   These imports serve as dependency health checks — a missing package "
        f"will cause an immediate failure before any test runs.\n"
        f"2. Below the imports, write one minimal test stub per function listed above.\n"
        f"3. {format_rules}\n"
        f"4. Return ONLY the complete test file content. "
        f"No markdown fences, no explanation text."
    )


def _format_rules(language: str, framework: str) -> str:
    """Return language/framework-specific formatting instructions."""
    if language == "Python":
        return (
            "Use pytest format: `def test_<name>():` with "
            "`raise NotImplementedError('<name> has no test coverage')`"
        )
    if language in ("JavaScript", "JavaScript/React"):
        return (
            "Use Jest format: `describe('<module>', () => { "
            "test('<name>', () => { throw new Error('<name> needs a test'); }); })`"
        )
    if language in ("TypeScript", "TypeScript/React"):
        return (
            "Use Jest/TypeScript format with `describe`/`test` blocks. "
            "Throw an Error in each stub body."
        )
    if language == "Rust":
        return (
            "Use Rust's `#[cfg(test)] mod tests { #[test] fn test_<name>() { "
            "panic!(\"<name> has no test coverage\"); } }` format."
        )
    if language == "Go":
        return (
            "Use Go testing format: `func Test<Name>(t *testing.T) { "
            "t.Skip(\"<Name> has no test coverage\") }` "
            "in a `package <pkg>_test` at the top."
        )
    return "Write appropriate test stubs for the detected language and framework."


def _default_framework(language: str) -> str:
    """Fall back to the canonical framework for each language."""
    return {
        "Python":              "pytest",
        "JavaScript":          "Jest",
        "JavaScript/React":    "Jest",
        "TypeScript":          "Jest",
        "TypeScript/React":    "Jest",
        "Rust":                "cargo test",
        "Go":                  "go test",
    }.get(language, "pytest")


# ===========================================================================
# Groq Call
# ===========================================================================

def _call_groq(groq_client: Any, prompt: str, module_node: ModuleNode) -> Optional[str]:
    """Call Groq and return the raw text response, or None on failure."""
    if groq_client is None:
        return None
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]  ⚠ Groq call failed for {Path(module_node.path).name}: "
            f"{exc}[/yellow]"
        )
        return None


# ===========================================================================
# Syntax Validation
# ===========================================================================

def _validate_syntax(content: str, language: str) -> tuple[bool, str]:
    """
    Validate generated code syntax.

    Returns (True, "") if valid.
    Returns (False, error_message) if invalid.
    If the validator tool is not available, returns (True, "") — we optimistically
    write the file and let the test runner surface errors naturally.
    """
    if language == "Python":
        try:
            ast.parse(content)
            return True, ""
        except SyntaxError as exc:
            return False, str(exc)

    if language in ("JavaScript", "JavaScript/React", "TypeScript", "TypeScript/React"):
        return _run_validator(["node", "--check"], content, ".js")

    if language == "Go":
        return _run_validator(["gofmt", "-e"], content, ".go")

    if language == "Rust":
        return _run_validator(["rustfmt", "--check"], content, ".rs")

    # Unknown language — optimistically accept
    return True, ""


def _run_validator(
    cmd_base: List[str],
    content: str,
    suffix: str,
) -> tuple[bool, str]:
    """
    Write content to a temp file, run the validator command, return result.
    If the command is not found, return (True, "") — tool not installed.
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, encoding="utf-8", delete=False
        ) as tf:
            tf.write(content)
            tmp_path = tf.name

        cmd = cmd_base + [tmp_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        Path(tmp_path).unlink(missing_ok=True)

        if result.returncode != 0:
            err = (result.stderr or result.stdout).strip()[:200]
            return False, err
        return True, ""

    except FileNotFoundError:
        # Validator binary not on PATH — skip validation
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return True, ""   # any other error → optimistically accept


# ===========================================================================
# Fallback Template
# ===========================================================================

def _make_fallback(
    module_node: ModuleNode,
    func_names: List[str],
    reason: str = "",
) -> str:
    """
    Generate a minimal syntactically-valid fallback test file.
    Written to disk when Groq output fails syntax validation.
    The file is valid enough for the test runner to execute and surface errors.
    """
    language   = module_node.language
    func_list  = ", ".join(func_names)
    header_msg = f"ContribTriage: AI-generated test file — {reason}" if reason else \
                 "ContribTriage: fallback stub placeholder"

    if language == "Python":
        stubs = "\n\n".join(
            f"def test_{fn}():\n"
            f'    """Fallback stub for {fn} — please implement."""\n'
            f'    raise NotImplementedError("{fn} has no test coverage.")'
            for fn in func_names
        )
        return (
            f'"""\n{header_msg}\nUncovered functions: {func_list}\n"""\n\n'
            f"import pytest\n\n{stubs}\n"
        )

    if language in ("JavaScript", "JavaScript/React"):
        stubs = "\n\n".join(
            f"  test('{fn}', () => {{\n"
            f"    throw new Error('{fn} has no test coverage.');\n"
            f"  }});"
            for fn in func_names
        )
        return (
            f"// {header_msg}\n// Uncovered functions: {func_list}\n\n"
            f"describe('Module stubs', () => {{\n{stubs}\n}});\n"
        )

    if language in ("TypeScript", "TypeScript/React"):
        stubs = "\n\n".join(
            f"  test('{fn}', (): void => {{\n"
            f"    throw new Error('{fn} has no test coverage.');\n"
            f"  }});"
            for fn in func_names
        )
        return (
            f"// {header_msg}\n// Uncovered functions: {func_list}\n\n"
            f"describe('Module stubs', () => {{\n{stubs}\n}});\n"
        )

    if language == "Go":
        stubs = "\n\n".join(
            f"func Test{fn.capitalize()}(t *testing.T) {{\n"
            f'    t.Skip("{fn} has no test coverage.")\n'
            f"}}"
            for fn in func_names
        )
        return (
            f"// {header_msg}\n// Uncovered functions: {func_list}\n\n"
            f"package stub_test\n\nimport \"testing\"\n\n{stubs}\n"
        )

    if language == "Rust":
        stubs = "\n\n".join(
            f"    #[test]\n    fn test_{fn}() {{\n"
            f'        panic!("{fn} has no test coverage.");\n'
            f"    }}"
            for fn in func_names
        )
        return (
            f"// {header_msg}\n// Uncovered functions: {func_list}\n\n"
            f"#[cfg(test)]\nmod tests {{\n{stubs}\n}}\n"
        )

    # Generic Python fallback for unknown languages
    return (
        f"# {header_msg}\n# Uncovered functions: {func_list}\n\n"
        f"import pytest\n\n"
        f"def test_placeholder():\n"
        f'    raise NotImplementedError("Stub: {func_list}")\n'
    )


# ===========================================================================
# Helpers
# ===========================================================================

def _strip_fences(content: str) -> str:
    """Remove accidental markdown code fences from LLM output."""
    # Remove ```python, ```go, ```rust etc. and closing ```
    content = re.sub(r"^```[a-zA-Z]*\n?", "", content.lstrip(), flags=re.MULTILINE)
    content = re.sub(r"^```\s*$", "", content, flags=re.MULTILINE)
    return content.strip()
