import os
import json
from hypergraphrag import HyperGraphRAG
from hypergraphrag.llm import huggingface_bge_embedding

# os.environ["OPENAI_API_KEY"] = "ADD YOUR OPENAI_API_KEY HERE"
# os.environ['HUGGINGFACE_API_KEY'] = "ADD YOUR HUGGINGFACE_API_KEY HERE"

rag = HyperGraphRAG(
    embedding_func=huggingface_bge_embedding,
    chunk_token_size=400,
    chunk_overlap_token_size=50,
    node2vec_params={
        "dimensions": 1024,
        "num_walks": 10,
        "walk_length": 40,
        "window_size": 2,
        "iterations": 3,
        "random_seed": 3,
    },
    working_dir=f"expr/example"
)

with open(f"example_contexts.json", mode="r", encoding='utf-8') as f:
    unique_contexts = json.load(f)
    
rag.insert(unique_contexts)