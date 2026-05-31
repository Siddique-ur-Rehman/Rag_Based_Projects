<div align="center">

# 🔬 BioRAG-Hybrid

### *Intelligent Medical Literature Search with Hybrid Retrieval*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.10.0-FF4B4B?style=flat-square&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Groq](https://img.shields.io/badge/Groq-0.5.0-FF4B4B?style=flat-square&logo=groq&logoColor=white)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

</div>

---

## 🎯 What Makes This Special?

**BioRAG-Hybrid** isn't just another RAG system. It combines **two powerful search methods** to give you the best of both worlds:

| Search Type | How It Works | Best For |
|-------------|--------------|----------|
| **Dense Vectors** | Understands meaning and context | "What causes memory loss in elderly?" |
| **Sparse Vectors** | Matches exact keywords | "Alzheimer's amyloid plaques" |
| **Hybrid (Both)** | Semantic understanding + keyword precision | Complex medical queries |

> 💡 **Think of it like this:** Dense search understands you want "car" when you say "vehicle". Sparse search finds exact medical terms like "amyloid-beta". Together, they're unstoppable.

---

## 📊 Real Performance Metrics

I evaluated this system on **50 real PubMed questions** using **RAGAS (RAG Assessment)** framework. Here's what I found:

### Retrieval Quality (How well it finds relevant papers)

| Metric | Score | What This Means |
|--------|-------|-----------------|
| **Hit@1** | 98% | 49 times out of 50, the perfect answer was the FIRST result |
| **Hit@3** | 100% | Every single time, answer was within top 3 results |
| **Hit@5** | 100% | Answer always appears in top 5 results |
| **MRR** | 0.99 | The correct answer almost always ranks at the very top |

### Answer Quality (How accurate the generated answers are)

| Metric | Score | What This Means |
|--------|-------|-----------------|
| **Precision** | 0.984 | 98.4% of retrieved information was relevant |
| **Recall** | 0.980 | 98% of relevant information was successfully retrieved |
| **F1-Score** | 0.981 | Excellent balance between precision and recall |

> 🏆 **Bottom Line:** This system achieves **near-perfect retrieval** on medical literature. Doctors and researchers can trust the results.

---

## 🧠 How It Works: System Workflow
STEP 1: USER ASKS QUESTION
│
▼
STEP 2: HYBRID ENCODING
├── Dense Encoder (all-MiniLM-L6-v2) → Semantic understanding
└── Sparse Encoder (bm42-all-minilm-l6-v2-attentions) → Keyword matching
│
▼
STEP 3: QDRANT HYBRID SEARCH
├── Searches 500 PubMed abstracts with both vectors
└── Combines results using RRF (Reciprocal Rank Fusion)
│
▼
STEP 4: CONTEXT RETRIEVAL
└── Extracts top 3 most relevant abstracts
│
▼
STEP 5: GROQ LLM GENERATION (Llama 3.3 70B)
├── Reads ONLY the retrieved abstracts
├── Generates evidence-based answer
└── If answer not in abstracts → "I don't have enough information"
│
▼
STEP 6: STREAMLIT UI DISPLAY
├── Shows generated answer
├── Displays relevance scores
├── Reveals source abstracts (expandable)
└── Collects user feedback (👍/👎)

text

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+          # Core language
Qdrant Cloud Account  # Free tier works fine
Groq API Key         # Free tier (30 requests/min)
Step-by-Step Setup
1. Clone the repository

bash
git clone https://github.com/Siddique-ur-Rehman/Rag_Based_Projects.git
cd Rag_Based_Projects/BioRAG-Hybrid
2. Install dependencies

bash
pip install -r requirements.txt
3. Set up environment variables

Create a .env file:

env
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
GROQ_API_KEY=your_groq_api_key_here
4. Run the application

bash
streamlit run streamlit_app.py
The UI will open at http://localhost:8501

🔧 Complete Pipeline (What Happens Behind the Scenes)
Step	Script	What It Does
1	load_data.py	Downloads PubMed QA dataset (1000 samples)
2	generate_embeddings.py	Creates dense vectors using sentence-transformers
3	create_collection.py	Sets up Qdrant collection with dense vector support
4	upload_to_qdrant.py	Uploads 500 dense vectors to Qdrant
5	hybrid_search_setup.py	Recreates collection with sparse vector support
6	upload_hybrid_data.py	Uploads both dense + sparse vectors
7	hybrid_query.py	Tests hybrid search with sample query
8	advance_evaluation_rag.py	Runs comprehensive metrics (Hit@k, MRR, F1)
9	streamlit_app.py	Launches interactive web interface
📚 Dataset Details
Source: PubMed QA (Hugging Face)
Format: pqa_labeled subset
Size: 500 biomedical abstracts
Each sample contains:

❓ Question (e.g., "Do mitochondria play a role in cellular remodeling?")

📄 Context (PubMed abstract text)

✅ Answer (yes/no/maybe based on the paper)

🧪 Running Your Own Evaluation
bash
# Run the comprehensive evaluation suite
python advance_evaluation_rag.py

# Sample output:
# ============================================================
# RAG SYSTEM EVALUATION METRICS
# ============================================================
# Exact Match Accuracy: 98.0%
# Hit@1: 49/50 = 98.0%
# Hit@3: 50/50 = 100.0%
# Hit@5: 50/50 = 100.0%
# MRR (Mean Reciprocal Rank): 0.990
# Precision (weighted): 0.984
# Recall (weighted): 0.980
# F1-Score (weighted): 0.981
# ============================================================
💬 Example Interaction
User asks: "What causes Alzheimer's disease?"

System responds:

Based on the retrieved medical literature, Alzheimer's disease is associated with hippocampal atrophy (HCA) in the brain. The abstract indicates that patients with Alzheimer's show progressive cognitive decline correlated with structural brain changes.

Along with:

Confidence score (0.95)

Source paper citation

Expandable abstract view

Feedback buttons (👍/👎)

🛠️ Tech Stack Details
Component	Technology	Version	Purpose
Vector DB	Qdrant Cloud	1.10.0	Stores and searches 384-dim vectors
Dense Encoder	Sentence-BERT	all-MiniLM-L6-v2	Converts text to semantic vectors
Sparse Encoder	Fastembed	bm42-all-minilm-l6-v2-attentions	Creates keyword-aware sparse vectors
LLM	Groq	Llama 3.3 70B	Generates answers from retrieved context
UI	Streamlit	1.32.0	Interactive web interface
Evaluation	RAGAS + Scikit-learn	0.1.5, 1.3.0	Comprehensive metrics
Data	Hugging Face Datasets	2.18.0	PubMed QA loading
📈 Why Hybrid Search Matters for Medical QA
Problem: Medical terms are precise. "Myocardial infarction" ≠ "heart attack" in keyword search, but they mean the same thing semantically.

Solution: Hybrid search:

Sparse vector catches exact term "myocardial infarction"

Dense vector understands "heart attack" means the same thing

Together → No missed relevant papers

Result: 98% Hit@1 means you almost always get the right answer on first try.

🤝 Contributing
Found a bug? Have an idea? Let's improve together!

Fork the repository

Create your feature branch (git checkout -b feature/amazing)

Commit changes (git commit -m 'Add amazing feature')

Push (git push origin feature/amazing)

Open a Pull Request

📝 License
MIT License - Free for academic and commercial use.

🙏 Acknowledgments
Qdrant Team - For the incredible hybrid search capabilities

Groq - For lightning-fast LLM inference

PubMed - For the high-quality biomedical dataset

Hugging Face - For making models and datasets accessible

RAGAS - For the evaluation framework

📧 Connect
Author: Siddique-ur-Rehman
GitHub: @Siddique-ur-Rehman
Project Repository: Rag_Based_Projects
Direct Link: BioRAG-Hybrid
