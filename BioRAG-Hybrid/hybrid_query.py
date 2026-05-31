from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

dense_model = SentenceTransformer("all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

query = "What causes Alzheimer's disease?"

dense_vec = dense_model.encode(query).tolist()
sparse_vec = list(sparse_model.embed(query))[0]

results = client.query_points(
    collection_name="pubmed_hybrid",
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=10),
        Prefetch(query=sparse_vec.as_object(), using="sparse", limit=10)
    ],
    query=FusionQuery(fusion="rrf"),
    limit=5
)

for res in results.points:
    print(f"Score: {res.score}")
    print(f"Context: {res.payload['context'][:200]}...")
    print(f"Answer: {res.payload['answer']}\n")