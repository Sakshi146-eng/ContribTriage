import tree_sitter_javascript as tsjs
import tree_sitter_rust as tsrs
from tree_sitter import Language, Parser

# ── JS imports ────────────────────────────────────────────────────────────
js_lang = Language(tsjs.language())
p = Parser(js_lang)
src = b"import React from 'react';\nimport { useState } from 'react';\nconst x = require('axios');\n"
tree = p.parse(src)
q = js_lang.query("(import_statement) @stmt")
stmts = [n.text.decode() for n in q.captures(tree.root_node).get("stmt", [])]
print("JS import stmts:", stmts)

# Try getting the source string directly
q2 = js_lang.query('(import_statement source: (string (string_fragment) @src))')
srcs = [n.text.decode() for n in q2.captures(tree.root_node).get("src", [])]
print("JS import sources:", srcs)

# ── Rust use ─────────────────────────────────────────────────────────────
rs_lang = Language(tsrs.language())
p2 = Parser(rs_lang)
src2 = b"use std::collections::HashMap;\nuse crate::models::LangGraphState;\nextern crate serde;\n"
tree2 = p2.parse(src2)
q3 = rs_lang.query("(use_declaration) @stmt")
stmts2 = [n.text.decode() for n in q3.captures(tree2.root_node).get("stmt", [])]
print("Rust use stmts:", stmts2)

q4 = rs_lang.query("(extern_crate_declaration name: (identifier) @name)")
extern_names = [n.text.decode() for n in q4.captures(tree2.root_node).get("name", [])]
print("Rust extern crates:", extern_names)

# ── Python: verify async def captured ────────────────────────────────────
import tree_sitter_python as tspy
py_lang = Language(tspy.language())
p3 = Parser(py_lang)
src3 = b"async def my_async_fn(): pass\ndef sync_fn(): pass\n"
tree3 = p3.parse(src3)
q5 = py_lang.query("(function_definition name: (identifier) @name)")
py_fns = [n.text.decode() for n in q5.captures(tree3.root_node).get("name", [])]
print("Python async+sync funcs:", py_fns)

# ── Go: type declarations ─────────────────────────────────────────────────
import tree_sitter_go as tsgo
go_lang = Language(tsgo.language())
p4 = Parser(go_lang)
src4 = b"type Handler struct{}\ntype Router interface{}\n"
tree4 = p4.parse(src4)
q6 = go_lang.query("(type_spec name: (type_identifier) @name)")
go_types = [n.text.decode() for n in q6.captures(tree4.root_node).get("name", [])]
print("Go types:", go_types)
