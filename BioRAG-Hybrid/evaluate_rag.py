from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client.models import Prefetch, FusionQuery
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

dense_model = SentenceTransformer("all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

with open("pubmed_data.json", "r") as f:
    data = json.load(f)

test_queries = data[:20]  # Test on first 20

results = []
for i, item in enumerate(test_queries):
    query = item["question"]
    ground_truth = item["answer"]
    print(f"Evaluating {i+1}/20: {query[:50]}...", flush=True)
    
    dense_vec = dense_model.encode(query).tolist()
    sparse_vec = list(sparse_model.embed(query))[0]
    
    response = client.query_points(
        collection_name="pubmed_hybrid",
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=5),
            Prefetch(query=sparse_vec.as_object(), using="sparse", limit=5)
        ],
        query=FusionQuery(fusion="rrf"),
        limit=3
    )
    
    contexts = [res.payload["context"] for res in response.points]
    retrieved_answers = [res.payload["answer"] for res in response.points]
    
    results.append({
        "question": query,
        "ground_truth": ground_truth,
        "retrieved_contexts": contexts,
        "retrieved_answers": retrieved_answers,
        "top_answer": retrieved_answers[0] if retrieved_answers else ""
    })

# Simple accuracy: matches ground truth with top retrieved answer
correct = sum(1 for r in results if r["ground_truth"].lower() == r["top_answer"].lower())
print(f"\n{'='*50}")
print(f"Simple Accuracy (top-1 match): {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
print(f"{'='*50}")

# Save results
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved evaluation_results.json")