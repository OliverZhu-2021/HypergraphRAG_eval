import os
from hypergraphrag import HyperGraphRAG
os.environ["OPENAI_API_KEY"] = "sk-YPr0hlW1DGGSo9My0c2dB036C0414bEe96D6Ea7a32945cC4"

rag = HyperGraphRAG(working_dir=f"expr/example")

query_text = 'How strong is the evidence supporting a systolic BP target of 120-129 mmHg in elderly or frail patients, considering potential risks like orthostatic hypotension, the balance between cardiovascular benefits and adverse effects, and the feasibility of implementation in diverse healthcare settings?'

result = rag.query(query_text)
print(result)