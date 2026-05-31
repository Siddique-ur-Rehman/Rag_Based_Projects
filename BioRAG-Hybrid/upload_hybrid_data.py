from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import SparseTextEmbedding
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

with open("pubmed_data_with_vectors.json", "r") as f:
    data = json.load(f)

# Use supported model with ONNX
sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

points = []
for idx, item in enumerate(data):
    if idx % 100 == 0:
        print(f"Processing {idx}/500", flush=True)
    sparse_vec = list(sparse_model.embed(item["context"]))[0]
    
    points.append(PointStruct(
        id=idx,
        vector={
            "dense": item["dense_vector"],
            "sparse": sparse_vec.as_object()
        },
        payload={
            "question": item["question"],
            "context": item["context"],
            "answer": item["answer"]
        }
    ))

client.upsert(collection_name="pubmed_hybrid", points=points)
print(f"Uploaded {len(points)} points with hybrid vectors")