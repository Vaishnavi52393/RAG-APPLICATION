"""
vector_store.py - Vector Database Layer powered by ChromaDB & Google GenAI Embeddings.
Part of the Cloud-Based RAG Document Assistant.

Provides:
- Asymmetric embedding creation with Google GenAI SDK (gemini-embedding-2-preview / text-embedding-004)
- ChromaDB vector collection storage with metadata persistence
- Semantic similarity search (cosine distance / similarity scoring)
- Document deletion, collection stats, and metadata filtering
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# Google GenAI modern SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore

# ChromaDB
try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except ImportError:
    chromadb = None  # type: ignore
    Collection = Any  # type: ignore

from ingest import DocumentChunk


DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2-preview"
FALLBACK_EMBEDDING_MODEL = "text-embedding-004"


class GeminiEmbeddingFunction:
    """
    Chroma-compatible embedding function wrapping Google GenAI SDK.
    Handles rate-limiting batches and asymmetric task types.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Set it in your environment or .env file."
            )
        self.model_name = model_name
        self.task_type = task_type
        if genai is None:
            raise ImportError(
                "google-genai SDK not installed. Run 'pip install google-genai'."
            )
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"headers": {"User-Agent": "aistudio-build-rag"}},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of documents (chunks)."""
        return self._embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> List[float]:
        """Compute asymmetric embedding for a single user search query."""
        results = self._embed_batch([query], task_type="RETRIEVAL_QUERY")
        return results[0] if results else []

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Callable interface expected by ChromaDB."""
        return self.embed_documents(input)

    def _embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Internal batching logic to respect API limits."""
        embeddings: List[List[float]] = []
        batch_size = 16  # Safe batch size for embeddings

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                # Primary attempt with configured model
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type) if types else None,
                )
                if hasattr(response, "embeddings") and response.embeddings:
                    for emb in response.embeddings:
                        embeddings.append(list(emb.values))
                elif hasattr(response, "embedding") and response.embedding:
                    embeddings.append(list(response.embedding.values))
            except Exception as e:
                # Fallback to text-embedding-004 if preview model is unavailable
                try:
                    fallback_resp = self.client.models.embed_content(
                        model=FALLBACK_EMBEDDING_MODEL,
                        contents=batch,
                    )
                    if hasattr(fallback_resp, "embeddings") and fallback_resp.embeddings:
                        for emb in fallback_resp.embeddings:
                            embeddings.append(list(emb.values))
                except Exception as fallback_err:
                    raise RuntimeError(
                        f"Failed to generate embeddings via Gemini API: {e} | Fallback failed: {fallback_err}"
                    )

        return embeddings


class VectorStore:
    """
    Manages ChromaDB collections, chunk vector indexing, and semantic similarity search.
    """

    def __init__(
        self,
        collection_name: str = "cloud_rag_documents",
        persist_directory: Optional[str] = "./chroma_db",
        api_key: Optional[str] = None,
    ):
        if chromadb is None:
            raise ImportError(
                "chromadb is not installed. Run 'pip install chromadb'."
            )

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        # Initialize ChromaDB persistent client or Ephemeral client
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.chroma_client = chromadb.EphemeralClient()

        # Initialize Embedding Function
        self.embedding_fn = GeminiEmbeddingFunction(
            api_key=self.api_key,
            model_name=DEFAULT_EMBEDDING_MODEL,
        )

        # Get or create collection
        self.collection: Collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Embeds and stores document chunks with associated metadata in ChromaDB.
        Returns count of newly indexed chunks.
        """
        if not chunks:
            return 0

        # Filter out empty texts
        valid_chunks = [c for c in chunks if c.text and c.text.strip()]
        if not valid_chunks:
            return 0

        texts = [c.text for c in valid_chunks]
        ids = [c.chunk_id for c in valid_chunks]
        metadatas = [c.to_dict() for c in valid_chunks]

        # Clean metadata to primitive types accepted by Chroma (str, int, float, bool)
        sanitized_metadatas: List[Dict[str, Any]] = []
        for m in metadatas:
            clean_meta = {}
            for k, v in m.items():
                if k == "text":
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            sanitized_metadatas.append(clean_meta)

        # Compute document embeddings
        embeddings = self.embedding_fn.embed_documents(texts)

        # Upsert into Chroma collection
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=sanitized_metadatas,
        )
        return len(valid_chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_similarity: float = 0.0,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic nearest-neighbor search for a user query.
        Returns sorted list of matches with similarity scores, text, and metadata.
        """
        if self.collection.count() == 0:
            return []

        # Generate asymmetric query embedding
        query_embedding = self.embedding_fn.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        matched_chunks: List[Dict[str, Any]] = []

        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

            for doc_text, meta, distance, cid in zip(docs, metas, dists, ids):
                # Cosine distance in Chroma: dist in [0, 2], where 0 is identical
                # Convert distance to similarity score in [0, 1]
                similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

                if similarity >= min_similarity:
                    matched_chunks.append({
                        "chunk_id": cid or meta.get("chunk_id", "unknown"),
                        "text": doc_text,
                        "source": meta.get("source", "unknown"),
                        "page_number": meta.get("page_number", 1),
                        "section_index": meta.get("section_index", 0),
                        "char_count": meta.get("char_count", len(doc_text)),
                        "similarity_score": round(similarity, 4),
                        "distance": round(distance, 4),
                        "metadata": meta,
                    })

        # Sort descending by similarity
        matched_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matched_chunks

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about indexed documents and chunks."""
        count = self.collection.count()
        sources = set()

        if count > 0:
            # Peek up to 1000 items to gather unique sources
            peek_data = self.collection.get(limit=1000, include=["metadatas"])
            if peek_data and peek_data.get("metadatas"):
                for m in peek_data["metadatas"]:
                    if m and "source" in m:
                        sources.add(m["source"])

        return {
            "total_chunks": count,
            "total_documents": len(sources),
            "sources": sorted(list(sources)),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_fn.model_name,
            "distance_metric": "cosine",
            "persist_directory": self.persist_directory,
        }

    def clear(self) -> None:
        """Deletes all chunks and recreates the collection."""
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
