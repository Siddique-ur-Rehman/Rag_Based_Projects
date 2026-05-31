from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client.models import Prefetch, FusionQuery
import json
import os
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

dense_model = SentenceTransformer("all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

with open("pubmed_data.json", "r") as f:
    data = json.load(f)

test_queries = data[:50]

y_true = []
y_pred = []
hit_at_1 = 0
hit_at_3 = 0
hit_at_5 = 0
mrr = 0

for i, item in enumerate(test_queries):
    query = item["question"]
    ground_truth = item["answer"]
    
    dense_vec = dense_model.encode(query).tolist()
    sparse_vec = list(sparse_model.embed(query))[0]
    
    response = client.query_points(
        collection_name="pubmed_hybrid",
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=10),
            Prefetch(query=sparse_vec.as_object(), using="sparse", limit=10)
        ],
        query=FusionQuery(fusion="rrf"),
        limit=5
    )
    
    retrieved_answers = [res.payload["answer"] for res in response.points]
    
    # Exact match accuracy
    y_true.append(ground_truth)
    y_pred.append(retrieved_answers[0] if retrieved_answers else "")
    
    # Hit@k
    if ground_truth in retrieved_answers[:1]:
        hit_at_1 += 1
    if ground_truth in retrieved_answers[:3]:
        hit_at_3 += 1
    if ground_truth in retrieved_answers[:5]:
        hit_at_5 += 1
    
    # MRR
    for rank, ans in enumerate(retrieved_answers[:5], 1):
        if ans == ground_truth:
            mrr += 1 / rank
            break

# Calculate metrics
exact_match = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
precision = precision_score(y_true, y_pred, average='weighted', pos_label='yes')
recall = recall_score(y_true, y_pred, average='weighted', pos_label='yes')
f1 = f1_score(y_true, y_pred, average='weighted', pos_label='yes')
mrr_score = mrr / len(test_queries)

print("\n" + "="*60)
print("RAG SYSTEM EVALUATION METRICS")
print("="*60)
print(f"Exact Match Accuracy: {exact_match*100:.1f}%")
print(f"Hit@1: {hit_at_1}/{len(test_queries)} = {hit_at_1/len(test_queries)*100:.1f}%")
print(f"Hit@3: {hit_at_3}/{len(test_queries)} = {hit_at_3/len(test_queries)*100:.1f}%")
print(f"Hit@5: {hit_at_5}/{len(test_queries)} = {hit_at_5/len(test_queries)*100:.1f}%")
print(f"MRR (Mean Reciprocal Rank): {mrr_score:.3f}")
print(f"Precision (weighted): {precision:.3f}")
print(f"Recall (weighted): {recall:.3f}")
print(f"F1-Score (weighted): {f1:.3f}")
print("="*60)