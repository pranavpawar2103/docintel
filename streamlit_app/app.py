"""
Streamlit Frontend for DocIntel
Beautiful web interface for document Q&A
"""

import streamlit as st
import requests
from pathlib import Path
import time
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="DocIntel - Document Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_statistics():
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats")
        return response.json()
    except Exception as e:
        st.error(f"Failed to get statistics: {str(e)}")
        return None

def list_documents():
    """Get list of documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/documents")
        return response.json()
    except Exception as e:
        st.error(f"Failed to list documents: {str(e)}")
        return None

def upload_document(file):
    """Upload a document."""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(
            f"{API_BASE_URL}/api/documents/upload",
            files=files
        )
        return response.json()
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return None

def query_documents(question, n_results=5):
    """Ask a question."""
    try:
        data = {
            "question": question,
            "n_results": n_results,
            "include_context": False
        }
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json=data
        )
        return response.json()
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return None

def delete_document(doc_id):
    """Delete a document."""
    try:
        response = requests.delete(f"{API_BASE_URL}/api/documents/{doc_id}")
        return response.json()
    except Exception as e:
        st.error(f"Delete failed: {str(e)}")
        return None

def get_confidence_color(confidence):
    """Get CSS class for confidence level."""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def format_confidence(confidence):
    """Format confidence score."""
    percentage = confidence * 100
    if confidence >= 0.7:
        emoji = "🟢"
    elif confidence >= 0.5:
        emoji = "🟡"
    else:
        emoji = "🔴"
    return f"{emoji} {percentage:.1f}%"

# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">📚 DocIntel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Intelligent Document Analysis with RAG</div>',
        unsafe_allow_html=True
    )
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not running! Please start the backend first:")
        st.code("uvicorn src.api.main:app --reload --port 8000")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Statistics
        stats = get_statistics()
        if stats:
            st.subheader("📊 System Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats['total_documents'])
                st.metric("Queries", stats['total_queries'])
            with col2:
                st.metric("Chunks", stats['total_chunks'])
                uptime_min = stats.get('uptime_seconds', 0) / 60
                st.metric("Uptime", f"{uptime_min:.1f}m")
        
        st.divider()
        
        # Document Management
        st.subheader("📄 Document Management")
        
        # Upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'docx', 'txt', 'md'],
            help="Upload a document to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Upload & Process"):
                with st.spinner("Processing document..."):
                    result = upload_document(uploaded_file)
                    
                    if result and result.get('success'):
                        st.success(f"✅ {result['message']}")
                        st.info(f"Document ID: `{result['document_id']}`")
                        st.info(f"Chunks created: {result['num_chunks']}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Upload failed")
        
        st.divider()
        
        # List documents
        st.subheader("📚 Indexed Documents")
        docs = list_documents()
        
        if docs and docs['documents']:
            for doc_id in docs['documents']:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(doc_id)
                with col2:
                    if st.button("🗑️", key=f"del_{doc_id}"):
                        result = delete_document(doc_id)
                        if result:
                            st.success("Deleted!")
                            time.sleep(0.5)
                            st.rerun()
        else:
            st.info("No documents indexed yet")
    
    # Main content area
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📖 About"])
    
    with tab1:
        # Query Interface
        st.header("Ask Questions About Your Documents")
        
        # Query input
        question = st.text_input(
            "Your Question:",
            placeholder="What is machine learning?",
            help="Ask any question about your indexed documents"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            n_results = st.slider(
                "Number of sources to retrieve",
                min_value=1,
                max_value=10,
                value=5,
                help="How many document chunks to use for answering"
            )
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            query_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
# Process query
        if query_button and question:
            with st.spinner("Thinking..."):
                start_time = time.time()
                result = query_documents(question, n_results)
                elapsed = time.time() - start_time
                
                # Check if result is valid
                if result is None:
                    st.error("❌ Failed to get response from API. Please check if the backend is running.")
                elif 'answer' not in result:
                    st.error(f"❌ Invalid response from API: {result}")
                else:
                    # Display answer
                    st.subheader("💡 Answer")
                    st.write(result['answer'])
                    
                    # Display metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Confidence",
                            format_confidence(result.get('confidence', 0.0))
                        )
                    with col2:
                        st.metric("Processing Time", f"{result.get('processing_time_ms', 0)}ms")
                    with col3:
                        st.metric("Tokens Used", result.get('tokens_used', 0))
                    with col4:
                        st.metric("Model", result.get('model', 'N/A'))
                    
                    # Display sources
                    if result.get('sources'):
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(
                                f"Source {i}: {source['document_name']} (Page {source['page_number']})"
                            ):
                                st.write(f"**Relevance Score:** {source['relevance_score']:.3f}")
                    else:
                        st.info("No sources found (answer may be based on insufficient context)")
        
        elif query_button:
            st.warning("Please enter a question")
        
        # Example questions
        st.divider()
        st.subheader("💡 Example Questions")
        examples = [
            "What is the main topic of the document?",
            "Can you summarize the key points?",
            "What methodology was used?",
            "What are the conclusions?"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}"):
                    st.session_state.example_question = example
                    st.rerun()
    
    with tab2:
        # About page
        st.header("About DocIntel")
        
        st.markdown("""
        ### What is DocIntel?
        
        DocIntel is an intelligent document analysis system powered by **Retrieval-Augmented Generation (RAG)**.
        
        ### How it works:
        
        1. **📤 Upload Documents**: Upload PDFs, Word documents, or text files
        2. **🔍 Intelligent Processing**: Documents are automatically:
           - Parsed and chunked intelligently
           - Converted to embeddings (vectors)
           - Indexed in a vector database
        3. **💬 Ask Questions**: Ask questions in natural language
        4. **✨ Get Answers**: Receive accurate answers with source citations
        
        ### Technology Stack:
        
        - **LLM**: OpenAI GPT-4o-mini
        - **Embeddings**: OpenAI text-embedding-3-small
        - **Vector Database**: ChromaDB
        - **Framework**: LangChain
        - **Backend**: FastAPI
        - **Frontend**: Streamlit
        
        ### Features:
        
        ✅ Multi-format support (PDF, DOCX, TXT, MD)  
        ✅ Intelligent text chunking  
        ✅ Semantic search  
        ✅ Citation tracking  
        ✅ Confidence scoring  
        ✅ Real-time processing  
        
        ### Created by: Pranav
        Master's in Computer Science, University of Ottawa
        """)
        
        st.divider()
        
        # System info
        if stats:
            st.subheader("🔧 System Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Documents:** {stats['total_documents']}")
                st.write(f"**Total Chunks:** {stats['total_chunks']}")
            with col2:
                st.write(f"**Queries Processed:** {stats['total_queries']}")
                uptime_min = stats.get('uptime_seconds', 0) / 60
                st.write(f"**Uptime:** {uptime_min:.1f} minutes")

if __name__ == "__main__":
    main()