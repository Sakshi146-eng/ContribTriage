from contribtriage.ingestion.lexical_parser import build_knowledge_graph

kg = build_knowledge_graph("c:/Users/saksh/OneDrive/Desktop/Apni Kaksha/ContribTriage")
print(f"Modules:   {len(kg.nodes)}")
print(f"Edges:     {len(kg.edges)}")
print(f"Uncovered: {len(kg.uncovered_funcs)}")
print(f"Languages: {kg.language_summary}")

for name, node in list(kg.nodes.items())[:8]:
    print(f"  {name} [{node.language}]: funcs={node.functions[:3]} imports={node.imports[:2]}")
