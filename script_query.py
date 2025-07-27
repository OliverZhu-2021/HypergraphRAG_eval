import os
from hypergraphrag import HyperGraphRAG
from hypergraphrag.llm import huggingface_bge_embedding

os.environ["OPENAI_API_KEY"] = ""     # Add before run
# os.environ['HUGGINGFACE_API_KEY'] = ""

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

query_text = 'How strong is the evidence supporting a systolic BP target of 120-129 mmHg in elderly or frail patients, considering potential risks like orthostatic hypotension, the balance between cardiovascular benefits and adverse effects, and the feasibility of implementation in diverse healthcare settings?'

result = rag.query(query_text)
print(result)