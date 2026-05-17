import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load environment variables from .env file
load_dotenv()

print("=" * 50)
print("INGESTING DATA INTO QDRANT CLOUD")
print("=" * 50)

# Get Qdrant Cloud credentials from .env
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fbise_rules")

# Check if credentials exist
if not QDRANT_URL or not QDRANT_API_KEY:
    print(" ERROR: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
    print("Please check your .env file and try again.")
    exit(1)

print(f"\n📡 Connecting to Qdrant Cloud: {QDRANT_URL}")
print(f" Collection name: {COLLECTION_NAME}")

# Step 1: Load document
print("\n[1/4] Loading document...")
loader = TextLoader("data/rules_data.txt", encoding="utf-8")
documents = loader.load()
print(f" Loaded {len(documents)} document")

# Step 2: Split into chunks
print("\n[2/4] Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)
print(f"✓ Created {len(chunks)} chunks")

# Step 3: Create embeddings
print("\n[3/4] Loading embedding model (this takes 1-2 minutes first time)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✓ Embedding model ready")

# Step 4: Create vector store in Qdrant Cloud
print("\n[4/4] Creating vector store in Qdrant Cloud...")
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME
)
print("✓ Vector store created successfully!")

print("\n" + "=" * 50)
print(f" SUCCESS! Ingested {len(chunks)} chunks into Qdrant Cloud")
print(f" Collection: {COLLECTION_NAME}")
print("=" * 50)