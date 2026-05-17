import streamlit as st
import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="FBISE Rules Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize components
@st.cache_resource
def initialize_retriever():
    # Qdrant Cloud credentials
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fbise_rules")
    
    # Embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to Qdrant Cloud - FIXED SYNTAX
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

@st.cache_resource
def initialize_llm():
    # Groq LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    return llm

# Search and answer function
def get_answer(question, retriever, llm):
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    
    # Combine contexts
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for FBISE (Federal Board of Intermediate & Secondary Education, Islamabad).
        
Answer the user's question based ONLY on the following context from the FBISE examination rules.

If the answer cannot be found in the context, say "I cannot find this information in the examination rules."

Be concise, accurate, and helpful.

CONTEXT:
{context}
"""),
        ("human", "{question}")
    ])
    
    # Generate answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return answer, docs

# Main app
def main():
    st.title("📚 FBISE Examination Rules Assistant")
    st.markdown("Ask any question about FBISE examination rules, policies, and procedures.")
    
    # Initialize
    with st.spinner("Loading assistant..."):
        retriever = initialize_retriever()
        llm = initialize_llm()
    
    st.success("✅ Assistant ready!")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This assistant can answer questions about:
        - Examination rules and policies
        - Grading system
        - Recounting procedures
        - Eligibility criteria
        - Fee refunds
        - Special accommodations
        - Change of subjects/groups
        - Improvement of grades
        """)
        
        st.header("📝 Example Questions")
        st.markdown("""
        - What is the minimum passing percentage?
        - How do I apply for recounting?
        - What are the rules for improvement of grades?
        - Can a private candidate appear for SSC Technical Group?
        - What is the policy for amanuensis/writer?
        """)
        
        st.divider()
        st.caption("Powered by Groq + Qdrant Cloud")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question about FBISE rules..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching rules and generating answer..."):
                answer, docs = get_answer(prompt, retriever, llm)
                st.markdown(answer)
                
                # Show sources
                with st.expander("📖 View sources"):
                    for i, doc in enumerate(docs, 1):
                        st.text_area(f"Source {i}", doc.page_content, height=150)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()