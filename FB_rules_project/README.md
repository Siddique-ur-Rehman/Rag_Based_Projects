# 📚 FBISE Examination Rules Assistant

An intelligent RAG (Retrieval-Augmented Generation) chatbot that answers questions about FBISE (Federal Board of Intermediate & Secondary Education, Islamabad) examination rules and policies.

## 🎯 Features

- 💬 **Interactive Chat Interface** - Ask questions naturally
- 🔍 **Semantic Search** - Finds relevant rules using vector embeddings
- 🤖 **AI-Powered Answers** - Uses Groq's Mixtral 8x7B LLM
- 📖 **Source Attribution** - Shows which rules were used to generate answers
- ⚡ **Fast Responses** - Powered by Qdrant vector database
- 🎨 **Clean UI** - Built with Streamlit

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework |
| **Qdrant Cloud** | Vector database for semantic search |
| **Groq** | LLM inference (Mixtral 8x7B) |
| **LangChain** | RAG pipeline orchestration |
| **Sentence Transformers** | Embeddings (all-MiniLM-L6-v2) |

## 📋 Prerequisites

- Python 3.11+
- Qdrant Cloud account (free tier available)
- Groq API key (free tier available)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Siddique-ur-Rehman/Rag_Based_Projects.git
cd Rag_Based_Projects/FB_rules_project
