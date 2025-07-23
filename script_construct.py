import os
import json
from hypergraphrag import HyperGraphRAG
os.environ["OPENAI_API_KEY"] = "sk-YPr0hlW1DGGSo9My0c2dB036C0414bEe96D6Ea7a32945cC4"

rag = HyperGraphRAG(working_dir=f"expr/example")

with open(f"example_contexts.json", mode="r", encoding='utf-8') as f:
    unique_contexts = json.load(f)
    
rag.insert(unique_contexts)