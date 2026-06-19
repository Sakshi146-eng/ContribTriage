from contribtriage.runners.test_generator import _strip_fences, _validate_syntax

# Test strip_fences
inp = "```python\nimport os\n```"
result = _strip_fences(inp)
print(repr(result))
print("expected:", repr("import os"))
print("match:", result == "import os")

# Test validate_syntax
ok, msg = _validate_syntax("import os\ndef foo(): pass\n", "Python")
print(f"valid python: ok={ok}, msg={msg!r}")
ok2, msg2 = _validate_syntax("def invalid syntax !!!", "Python")
print(f"invalid python: ok={ok2}, msg={msg2!r}")
