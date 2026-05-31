from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, SparseIndexParams, Distance
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Delete existing collection
try:
    client.delete_collection("pubmed_hybrid")
    print("Deleted old collection")
except:
    print("No collection to delete")

# Create new with hybrid support
client.create_collection(
    collection_name="pubmed_hybrid",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams())
    }
)
print("✅ Hybrid collection created (dense + sparse)")