"""Quick check that all names imported by test_stage1_ingestion.py exist in lexical_parser."""
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
print("All test_stage1 imports: OK")
print("LANGUAGE_CONFIGS keys:", list(LANGUAGE_CONFIGS.keys()))

# Verify _parse_file works on each fixture
from pathlib import Path
FIXTURES = Path("tests/fixtures")
for ext, name in [(".py","sample.py"), (".ts","sample.ts"), (".rs","sample.rs"), (".go","sample.go")]:
    cfg = LANGUAGE_CONFIGS[ext]
    node = _parse_file(FIXTURES / name, f"sample_{ext.lstrip('.')}", cfg)
    print(f"  {ext}: {len(node.functions)} funcs, {len(node.classes)} classes, {len(node.imports)} imports, lang={node.language!r}")
print("All _parse_file checks: OK")
