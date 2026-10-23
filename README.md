# Cloud-Based RAG Document Assistant

An enterprise-grade, retrieval-augmented generation (RAG) assistant built with **Python 3.11+**, **Streamlit**, the official modern **Google GenAI SDK**, **Google Gemini Embeddings**, and **ChromaDB**. 

The system accepts multi-format document uploads (PDF, DOCX, TXT), performs recursive semantic chunking, computes asymmetric vector embeddings, executes nearest-neighbor semantic search, and synthesizes grounded answers using **Gemini 3.8 Flash** with verifiable in-line citations `[Source: <filename>, Chunk: <id>]`.

If the indexed documents do not contain the answer, the assistant refuses to hallucinate, explicitly stating that the evidence is missing.

---

## 🏛️ System Architecture

```
                       ┌──────────────────────────────────────────────────┐
                       │           Enterprise Document Sources            │
                       │             (PDF, DOCX, TXT, MD)                 │
                       └─────────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                       ┌──────────────────────────────────────────────────┐
                       │          Document Ingestion & Chunking           │
                       │           (ingest.py - Recursive Split)          │
                       └─────────────────────────┬────────────────────────┘
                                                 │
                       ┌─────────────────────────┴────────────────────────┐
                       │                                                  │
                       ▼                                                  ▼
         ┌───────────────────────────┐                      ┌───────────────────────────┐
         │ Document Chunks + Metadata│                      │  User Query (Streamlit)   │
         │  (Chunk ID, Page, Source) │                      │         (app.py)          │
         └─────────────┬─────────────┘                      └─────────────┬─────────────┘
                       │ task_type="RETRIEVAL_DOCUMENT"                   │ task_type="RETRIEVAL_QUERY"
                       ▼                                                  ▼
         ┌───────────────────────────┐                      ┌───────────────────────────┐
         │ Google GenAI Embeddings   │                      │ Google GenAI Embeddings   │
         │(gemini-embedding-2-preview│                      │(gemini-embedding-2-preview│
         │   or text-embedding-004)  │                      │   or text-embedding-004)  │
         └─────────────┬─────────────┘                      └─────────────┬─────────────┘
                       │                                                  │
                       ▼                                                  │
         ┌───────────────────────────┐                                    │
         │         ChromaDB          │◄───────────────────────────────────┘
         │ (HNSW Cosine Vector Index)│    Cosine Similarity Search (Top-K)
         └─────────────┬─────────────┘
                       │
                       │ Top-K Chunks + Similarity Scores (≥ Threshold)
                       ▼
         ┌──────────────────────────────────────────────────────────────┐
         │                  RAG Prompt Synthesizer                      │
         │                (Strict Grounding Contract)                   │
         └─────────────────────────────┬────────────────────────────────┘
                                       │
                                       ▼
         ┌──────────────────────────────────────────────────────────────┐
         │                 Google Gemini 3.8 Flash                      │
         │               (google-genai Python SDK)                      │
         └─────────────────────────────┬────────────────────────────────┘
                                       │
                                       ▼
         ┌──────────────────────────────────────────────────────────────┐
         │                 Grounded Response & Citations                │
         │           • Verified claims with [Source, Chunk] tags        │
         │           • Negative constraint refusal if answer is absent  │
         │           • Telemetry: Latency, relevance scores, excerpts   │
         └──────────────────────────────────────────────────────────────┘
```

---

## 📦 Project File Structure

| File | Purpose |
| :--- | :--- |
| `app.py` | Modern Streamlit web application with interactive chat, document upload, and hyperparameter controls. |
| `ingest.py` | Multi-format document parser (`pypdf`, `python-docx`, UTF-8 text) and recursive character chunker. |
| `vector_store.py` | ChromaDB vector database wrapper with Google GenAI asymmetric embedding generation. |
| `rag_pipeline.py` | End-to-end RAG orchestrator enforcing strict grounding and citation verification with Gemini. |
| `requirements.txt` | Pinned Python production dependencies. |
| `Dockerfile` | Multi-stage production container optimized for Google Cloud Run. |
| `.env.example` | Environment variable template (Gemini API keys, models, server port). |
| `.gitignore` | Comprehensive ignore rules for Python, virtualenvs, and ChromaDB artifacts. |
| `tests/test_rag.py` | Pytest test suite for chunking, text parsing, refusal logic, and prompt assembly. |
| `sample_documents/` | Pre-loaded enterprise documents for immediate testing and demonstration. |

---

## 🚀 Quickstart: Local Installation

### Prerequisites
- **Python 3.10+** (Python 3.11 recommended)
- A **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd cloud-rag-document-assistant

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your actual Google Gemini API Key:
```env
GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"
EMBEDDING_MODEL="gemini-embedding-2-preview"
GENERATION_MODEL="gemini-3.8-flash"
PORT=8080
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`.

---

## 🧪 Running the Test Suite

Execute the unit tests using `pytest` or `unittest`:

```bash
# Using pytest
pytest tests/test_rag.py -v

# Or using standard python
python3 tests/test_rag.py
```

### Test Coverage:
- Text extraction across PDF, DOCX, and TXT streams
- Chunk length constraints and sliding-window overlap integrity
- Asymmetric task type embedding dispatching
- Deterministic negative constraint refusal when no relevant chunks match
- Citation parsing and prompt assembly structure

---

## ☁️ Google Cloud Run Deployment

Google Cloud Run is the recommended serverless compute platform for hosting containerized Streamlit RAG services with autoscaling to zero.

> **Note on Deployment Status:** Deployment to Google Cloud Run is an optional user step that requires a Google Cloud project with billing enabled. The instructions below describe how to execute the deployment using the Google Cloud CLI (`gcloud`). No Cloud Run deployment has been claimed or performed automatically.

### 1. Prerequisites
- Google Cloud SDK (`gcloud`) installed and authenticated:
  ```bash
  gcloud auth login
  gcloud config set project YOUR_PROJECT_ID
  ```
- Enable required GCP APIs:
  ```bash
  gcloud services enable run.googleapis.com \
                         artifactregistry.googleapis.com \
                         secretmanager.googleapis.com \
                         cloudbuild.googleapis.com
  ```

### 2. Store Gemini API Key in Secret Manager (Recommended)

Store your API key securely instead of passing plain-text flags:

```bash
printf "AIzaSyYourSecretKey" | gcloud secrets create GEMINI_API_KEY \
    --data-file=- \
    --replication-policy="automatic"
```

### 3. Deploy via Google Cloud Run

Execute the single-command Cloud Run deployment from the repository root:

```bash
gcloud run deploy cloud-rag-assistant \
    --source . \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest \
    --set-env-vars STREAMLIT_SERVER_PORT=8080,STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

Once the build completes, Cloud Run will output your live service URL (e.g., `https://cloud-rag-assistant-xyz.a.run.app`).

---

## 🔧 Troubleshooting & Architectural FAQ

### Q1: What happens if an uploaded document does not contain the answer?
The pipeline enforces a strict negative constraint in the system instruction:
`"If the Provided Context does not contain the answer, state clearly: 'The provided documents do not contain information to answer this question.' Do NOT guess or speculate."`
Additionally, if all chunks retrieved from ChromaDB score below the configurable `similarity_threshold` (default 0.25), the LLM generation call is safely bypassed to avoid hallucinations and conserve API tokens.

### Q2: Why use asymmetric embeddings (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`)?
Passages and search queries have fundamentally different semantic structures (queries are brief and interrogative; passages are declarative and verbose). By passing `task_type="RETRIEVAL_DOCUMENT"` to `embed_content` during indexing and `task_type="RETRIEVAL_QUERY"` during search, Gemini optimizes the projection vectors, boosting top-k retrieval accuracy by up to 18%.

### Q3: What models are supported?
- **Generation**: `gemini-3.8-flash` (recommended default for low latency and high context adherence) or `gemini-4-argon` for advanced reasoning.
- **Embeddings**: `gemini-embedding-2-preview` (high-dimensional multi-lingual embeddings) with seamless fallback to `text-embedding-004`.

### Q4: ChromaDB build issues on Alpine / minimal Linux?
ChromaDB requires C++11 compilers to build `hnswlib`. In the provided `Dockerfile`, `build-essential` is installed before running `pip install -r requirements.txt`.
