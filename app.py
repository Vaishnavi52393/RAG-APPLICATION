"""
app.py - Streamlit Frontend for Cloud-Based RAG Document Assistant.
Enterprise Generative AI Retrieval-Augmented Generation (RAG) System.

Features:
- Multi-format file ingestion (PDF, DOCX, TXT)
- Dynamic chunking parameter tuning (Chunk Size, Overlap, Top-K)
- Semantic vector storage powered by ChromaDB
- Embeddings via Google GenAI SDK (gemini-embedding-2-preview / text-embedding-004)
- Generation via Gemini 3.8 Flash with strict grounding & in-line source citations
- Real-time telemetry: retrieval latency, cosine relevance scores, cited chunks
"""

import os
from pathlib import Path
import time
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# App imports
from ingest import process_document
from vector_store import VectorStore
from rag_pipeline import RagPipeline, NOT_FOUND_PHRASE

# Page Configuration
st.set_page_config(
    page_title="Cloud-Based RAG Document Assistant",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #5f6368;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .citation-box {
        background-color: #e8f0fe;
        border-left: 4px solid #1a73e8;
        padding: 10px;
        border-radius: 4px;
        font-size: 0.88rem;
        margin-top: 8px;
    }
    .grounding-verified {
        display: inline-block;
        background-color: #e6f4ea;
        color: #137333;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-bottom: 8px;
    }
    .grounding-refused {
        display: inline-block;
        background-color: #fce8e6;
        color: #c5221f;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def initialize_rag():
    """Initializes and caches the persistent vector store and RAG pipeline."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, None
    try:
        vstore = VectorStore(
            collection_name="cloud_rag_documents",
            persist_directory="./chroma_db",
            api_key=api_key,
        )
        pipeline = RagPipeline(vector_store=vstore, api_key=api_key)
        return vstore, pipeline
    except Exception as e:
        st.error(f"Error initializing RAG components: {e}")
        return None, None


# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []

vector_store, rag_pipeline = initialize_rag()

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://www.gstatic.com/images/branding/googlelogo/svg/googlelogo_clr_74x24px.svg",
        width=110,
    )
    st.title("RAG Controls")

    # API Key status
    has_api_key = bool(os.environ.get("GEMINI_API_KEY"))
    if has_api_key:
        st.success("Google Gemini API Key: Active", icon="✅")
    else:
        st.error("GEMINI_API_KEY is not set in environment or .env file.")
        manual_key = st.text_input("Enter GEMINI_API_KEY:", type="password")
        if manual_key:
            os.environ["GEMINI_API_KEY"] = manual_key
            st.rerun()

    st.markdown("---")
    st.subheader("1. Document Ingestion")

    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    # Sample Documents Ingestion Button
    sample_dir = Path("./sample_documents")
    if sample_dir.exists() and sample_dir.is_dir():
        sample_files = list(sample_dir.glob("*.txt"))
        if sample_files:
            if st.button("📥 Load Pre-configured Sample Docs", use_container_width=True):
                with st.spinner("Indexing enterprise sample documents into ChromaDB..."):
                    loaded_count = 0
                    for sf in sample_files:
                        text_content = sf.read_text(encoding="utf-8")
                        chunks = process_document(
                            filename=sf.name,
                            file_bytes_or_text=text_content,
                            chunk_size=500,
                            chunk_overlap=100,
                        )
                        if vector_store:
                            vector_store.add_chunks(chunks)
                            loaded_count += len(chunks)
                    st.success(f"Indexed {loaded_count} chunks from sample documents!")
                    st.rerun()

    # Process Uploaded Files
    if uploaded_files and st.button("⚡ Index Uploaded Documents", use_container_width=True):
        if not vector_store:
            st.error("Vector store is not initialized. Please verify your GEMINI_API_KEY.")
        else:
            with st.spinner(f"Parsing and embedding {len(uploaded_files)} document(s)..."):
                total_new_chunks = 0
                for uf in uploaded_files:
                    file_bytes = uf.read()
                    chunks = process_document(
                        filename=uf.name,
                        file_bytes_or_text=file_bytes,
                        chunk_size=st.session_state.get("cfg_chunk_size", 500),
                        chunk_overlap=st.session_state.get("cfg_chunk_overlap", 100),
                    )
                    added = vector_store.add_chunks(chunks)
                    total_new_chunks += added

                st.success(f"Successfully indexed {total_new_chunks} chunks!")
                st.rerun()

    st.markdown("---")
    st.subheader("2. Hyperparameters")
    chunk_size = st.slider("Chunk Size (characters)", 200, 1500, 500, step=50, key="cfg_chunk_size")
    chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 300, 100, step=25, key="cfg_chunk_overlap")
    top_k = st.slider("Top-K Retrieved Chunks", 1, 10, 4, key="cfg_top_k")
    similarity_thresh = st.slider("Similarity Threshold", 0.0, 0.9, 0.25, step=0.05, key="cfg_similarity_thresh")

    st.markdown("---")
    st.subheader("3. Vector Store Health")
    if vector_store:
        stats = vector_store.get_stats()
        st.metric("Total Indexed Chunks", stats["total_chunks"])
        st.metric("Indexed Documents", stats["total_documents"])
        if stats["sources"]:
            with st.expander("Indexed Sources"):
                for s in stats["sources"]:
                    st.write(f"- {s}")

        if st.button("🗑️ Reset Vector Database", type="secondary", use_container_width=True):
            vector_store.clear()
            st.session_state.messages = []
            st.success("Vector database reset successfully.")
            st.rerun()


# -------------------------------------------------------------
# MAIN CHAT AREA
# -------------------------------------------------------------
st.markdown('<div class="main-title">Cloud-Based RAG Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enterprise semantic search with Google Gemini 3.8 Flash, text embeddings, and strict source citation grounding.</div>',
    unsafe_allow_html=True,
)

# Suggested queries
col1, col2, col3, col4 = st.columns(4)
sample_query = None

with col1:
    if st.button("🏗️ Cloud Run Specs", use_container_width=True):
        sample_query = "What are the key architectural specifications and limits for Google Cloud Run?"
with col2:
    if st.button("📊 Tabular Chunking", use_container_width=True):
        sample_query = "What is the recommended best practice for chunking tabular and structured data?"
with col3:
    if st.button("🔒 Compliance Retention", use_container_width=True):
        sample_query = "What are the data retention requirements for vector indices and chat sessions under the compliance policy?"
with col4:
    if st.button("❓ Out-of-Domain Test", use_container_width=True):
        sample_query = "How do you bake a traditional San Francisco sourdough bread?"

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg and msg["meta"]:
            m = msg["meta"]
            if m.get("grounded"):
                st.markdown('<span class="grounding-verified">✓ Grounding Verified Against Context</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="grounding-refused">⚠ Refusal / No Source Match</span>', unsafe_allow_html=True)

            citations = m.get("citations", [])
            if citations:
                with st.expander(f"📚 View {len(citations)} Verified Source Citation(s)", expanded=False):
                    for c in citations:
                        score_pct = f"{c['similarity_score']:.1%}" if c.get("similarity_score") is not None else "N/A"
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <strong>Source:</strong> {c['source']} | <strong>Chunk:</strong> {c['chunk_id']} | <strong>Relevance:</strong> {score_pct}
                                <br/><em>"{c.get('excerpt', '')}"</em>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Telemetry footer
            ret_ms = m.get("retrieval_ms", 0)
            gen_ms = m.get("generation_ms", 0)
            st.caption(f"⚡ Retrieval: {ret_ms:.1f}ms | Generation: {gen_ms:.1f}ms | Model: {m.get('model', 'gemini-3.8-flash')}")

# Chat Input Prompt
user_input = st.chat_input("Ask a question about your uploaded documents...")
prompt_to_process = sample_query or user_input

if prompt_to_process:
    if not rag_pipeline or not vector_store:
        st.error("Please configure your GEMINI_API_KEY to start asking questions.")
    elif vector_store.get_stats()["total_chunks"] == 0:
        st.warning("Your vector store is empty! Please upload documents or click 'Load Pre-configured Sample Docs' in the sidebar first.")
    else:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving semantic chunks and synthesizing grounded answer with Gemini..."):
                resp = rag_pipeline.query(
                    user_query=prompt_to_process,
                    top_k=top_k,
                    min_similarity=similarity_thresh,
                )

            st.markdown(resp.answer)

            # Grounding badge
            if resp.grounded:
                st.markdown('<span class="grounding-verified">✓ Grounding Verified Against Context</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="grounding-refused">⚠ Refusal / No Source Match</span>', unsafe_allow_html=True)

            # Citations list
            if resp.citations:
                with st.expander(f"📚 View {len(resp.citations)} Verified Source Citation(s)", expanded=True):
                    for c in resp.citations:
                        score_pct = f"{c['similarity_score']:.1%}" if c.get("similarity_score") is not None else "N/A"
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <strong>Source:</strong> {c['source']} | <strong>Chunk:</strong> {c['chunk_id']} | <strong>Relevance:</strong> {score_pct}
                                <br/><em>"{c.get('excerpt', '')}"</em>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.caption(f"⚡ Retrieval: {resp.retrieval_ms:.1f}ms | Generation: {resp.generation_ms:.1f}ms | Model: {resp.model}")

            # Save in session
            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.answer,
                "meta": {
                    "grounded": resp.grounded,
                    "citations": resp.citations,
                    "retrieval_ms": resp.retrieval_ms,
                    "generation_ms": resp.generation_ms,
                    "model": resp.model,
                },
            })
