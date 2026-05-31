import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client.models import Prefetch, FusionQuery
from groq import Groq
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Page config
st.set_page_config(
    page_title="BioRAG-Hybrid",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00ff00;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🔬 BioRAG-Hybrid</h1>
    <p style="color: white; margin: 0;">Intelligent Medical Literature Search with Hybrid Retrieval</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-doctor.png", width=80)
    st.markdown("### ⚙️ Settings")
    
    search_mode = st.radio(
        "🔍 Search Mode",
        ["🎯 Hybrid (Dense + Sparse)", "📊 Dense Only"],
        help="Hybrid combines semantic understanding with keyword matching"
    )
    
    top_k = st.slider(
        "📚 Number of results",
        min_value=1,
        max_value=10,
        value=3,
        help="How many relevant papers to retrieve"
    )
    
    st.markdown("---")
    st.markdown("### 🤖 LLM Settings")
    
    temperature = st.slider(
        "🌡️ Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    st.markdown("---")
    st.markdown("### 📊 System Stats")
    st.info("✅ Collection: pubmed_hybrid\n✅ Model: all-MiniLM-L6-v2\n✅ Hybrid: Active")
    
    st.markdown("---")
    st.markdown("💡 **Example Questions:**")
    if st.button("🧠 Alzheimer's causes"):
        st.session_state.query = "What causes Alzheimer's disease?"
    if st.button("❤️ Heart disease risk"):
        st.session_state.query = "What are the risk factors for heart disease?"
    if st.button("💊 Diabetes treatment"):
        st.session_state.query = "What are effective treatments for diabetes?"

# Main content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("### 💬 Ask Your Medical Question")
    
    if "query" not in st.session_state:
        st.session_state.query = ""
    
    query = st.text_area(
        "",
        value=st.session_state.query,
        placeholder="e.g., What causes Alzheimer's disease?",
        height=100,
        label_visibility="collapsed"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        search_clicked = st.button("🔍 Search Medical Literature", use_container_width=True)

# Loading models
@st.cache_resource
def load_models():
    with st.spinner("🚀 Loading AI models..."):
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")
        llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return client, dense_model, sparse_model, llm

if search_clicked and query:
    try:
        with st.spinner("🔄 Processing your question..."):
            client, dense_model, sparse_model, llm = load_models()
            
            # Generate embeddings
            dense_vec = dense_model.encode(query).tolist()
            
            # Search
            if "Hybrid" in search_mode:
                sparse_vec = list(sparse_model.embed(query))[0]
                results = client.query_points(
                    collection_name="pubmed_hybrid",
                    prefetch=[
                        Prefetch(query=dense_vec, using="dense", limit=top_k),
                        Prefetch(query=sparse_vec.as_object(), using="sparse", limit=top_k)
                    ],
                    query=FusionQuery(fusion="rrf"),
                    limit=top_k
                )
            else:
                results = client.query_points(
                    collection_name="pubmed_hybrid",
                    using="dense",
                    query=dense_vec,
                    limit=top_k
                )
            
            # Prepare context
            context = "\n\n---\n\n".join([
                f"📄 Source {i+1}:\n{res.payload['context'][:800]}"
                for i, res in enumerate(results.points)
            ])
            
            # Update progress
            st.markdown("### 🤖 Generating Answer with Groq...")
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            progress_bar.empty()
            
            # Generate answer
            response = llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=temperature,
                messages=[
                    {"role": "system", "content": """You are a compassionate medical assistant named BioRAG.

CRITICAL RULES:
1. Answer based ONLY on the provided PubMed abstracts
2. If the answer is NOT explicitly stated in the context, say: "I don't have enough information in the medical literature to answer this question confidently."
3. Do NOT make up or infer information not present in the abstracts
4. Be honest about limitations: "The retrieved papers don't address this specific question."
5. Never use phrases like "based on general knowledge" or "typically" - only use provided text

Example responses:
- ✅ Found in context: "According to the abstract, Alzheimer's is associated with amyloid plaques..."
- ❌ Not found: "The retrieved medical literature does not contain information about this specific question."""},
                    {"role": "user", "content": f"""📚 **Retrieved Medical Literature:**\n{context}\n\n❓ **Question:** {query}\n\n---\n**Answer based ONLY on the abstracts above. If not found, state you don't have this information:**"""}
                ]
            )
            answer = response.choices[0].message.content
            
            # Display answer
            st.markdown("---")
            st.markdown("## 💡 Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            
            # Display metrics
            st.markdown("## 📊 Retrieval Quality")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                avg_score = sum(r.score for r in results.points) / len(results.points)
                st.metric("🎯 Avg Relevance Score", f"{avg_score:.2f}")
            
            with col_b:
                st.metric("📄 Retrieved Papers", len(results.points))
            
            with col_c:
                st.metric("🔍 Search Mode", "Hybrid" if "Hybrid" in search_mode else "Dense")
            
            with col_d:
                st.metric("🌡️ LLM Temp", f"{temperature}")
            
            # Display sources
            st.markdown("## 📚 Retrieved Sources")
            tabs = st.tabs([f"Source {i+1}" for i in range(len(results.points))])
            
            for i, (tab, res) in enumerate(zip(tabs, results.points)):
                with tab:
                    st.markdown(f"**Score:** {res.score:.3f}")
                    st.markdown(f"**Answer in paper:** {res.payload['answer']}")
                    st.markdown("**Context excerpt:**")
                    st.info(res.payload['context'][:500] + "...")
            
            # Feedback
            st.markdown("---")
            col_fb1, col_fb2, col_fb3 = st.columns([1, 2, 1])
            with col_fb2:
                st.markdown("### Was this helpful?")
                col_thumb1, col_thumb2, col_thumb3 = st.columns(3)
                with col_thumb1:
                    st.button("👍 Yes", use_container_width=True)
                with col_thumb2:
                    st.button("👎 No", use_container_width=True)
                with col_thumb3:
                    st.button("🔄 New Search", use_container_width=True)
                    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Please check your API keys and Qdrant connection.")

elif search_clicked and not query:
    st.warning("⚠️ Please enter a question first!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by Qdrant Hybrid Search + Groq LLM | BioRAG-Hybrid System</p>",
    unsafe_allow_html=True
)