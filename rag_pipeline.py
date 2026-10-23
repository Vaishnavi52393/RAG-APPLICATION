"""
rag_pipeline.py - RAG Orchestration, Context Synthesis, and Grounded Generation.
Part of the Cloud-Based RAG Document Assistant.

Uses Google GenAI SDK (`google-genai`) with model 'gemini-3.8-flash'.
Enforces strict grounding: refuses to speculate if evidence is missing,
and mandates in-line source citations: [Source: <filename>, Chunk: <id>].
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore

from vector_store import VectorStore


DEFAULT_GENERATION_MODEL = "gemini-3.8-flash"
NOT_FOUND_PHRASE = "The provided documents do not contain information to answer this question."

RAG_SYSTEM_INSTRUCTION = """You are the Cloud-Based RAG Document Assistant, an enterprise generative AI retrieval engine.
Your mission is to provide accurate, truthful answers based strictly and exclusively on the retrieved source documents provided in the context.

STRICT GROUNDING & CITATION RULES:
1. Answer the question ONLY using explicit facts directly stated in the Provided Context below.
2. If the Provided Context does not contain the answer, or if the documents are completely silent on the topic, state clearly and unambiguously:
   "The provided documents do not contain information to answer this question."
   Do NOT attempt to guess, extrapolate, fabricate facts, or rely on prior outside pre-training knowledge.
3. Every factual claim or summary point MUST be backed by an in-line citation formatted exactly as:
   [Source: <filename>, Chunk: <chunk_id>]
4. Maintain a clear, professional, executive architectural tone. Use bullet points or numbered lists where appropriate for readability.
"""


@dataclass
class RagResponse:
    """Encapsulates the generated answer, retrieved source citations, and execution telemetry."""
    query: str
    answer: str
    grounded: bool
    retrieved_chunks: List[Dict[str, Any]]
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    model: str = DEFAULT_GENERATION_MODEL


class RagPipeline:
    """
    Coordinates semantic vector retrieval from ChromaDB and grounded response
    generation through Google Gemini.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GENERATION_MODEL,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is missing. Please set it in .env or environment."
            )

        self.vector_store = vector_store or VectorStore(api_key=self.api_key)
        self.model_name = model_name

        if genai is None:
            raise ImportError("google-genai is not installed. Run 'pip install google-genai'.")

        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"headers": {"User-Agent": "aistudio-build-rag"}},
        )

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Assembles retrieved chunks into structured context blocks with explicit metadata."""
        if not retrieved_chunks:
            return "No relevant source chunks retrieved."

        blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "Unknown")
            cid = chunk.get("chunk_id", f"chunk_{idx}")
            page = chunk.get("page_number", 1)
            score = chunk.get("similarity_score", 0.0)
            text = chunk.get("text", "").strip()

            block = (
                f"--- BEGIN SOURCE CHUNK {idx} ---\n"
                f"Source: {source}\n"
                f"Chunk ID: {cid}\n"
                f"Page/Section: {page}\n"
                f"Semantic Relevance Score: {score:.2%}\n"
                f"Content:\n{text}\n"
                f"--- END SOURCE CHUNK {idx} ---"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def query(
        self,
        user_query: str,
        top_k: int = 4,
        min_similarity: float = 0.25,
        temperature: float = 0.2,
    ) -> RagResponse:
        """
        Executes end-to-end RAG:
        1. Semantic search against ChromaDB
        2. Strict refusal check if no chunks meet threshold
        3. Contextual prompt assembly
        4. Gemini generation with in-line source citations
        """
        t0 = time.perf_counter()

        # Step 1: Semantic Retrieval
        retrieved_chunks = self.vector_store.search(
            query=user_query,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        t_retrieved = time.perf_counter()
        retrieval_ms = (t_retrieved - t0) * 1000

        # Edge case: If no relevant context passes threshold, refuse immediately
        if not retrieved_chunks:
            return RagResponse(
                query=user_query,
                answer=f"{NOT_FOUND_PHRASE}\n\n*Reason: No documents in the index met the semantic relevance threshold ({min_similarity:.0%}) for your question.*",
                grounded=False,
                retrieved_chunks=[],
                citations=[],
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=0.0,
                model=self.model_name,
            )

        # Step 2: Assemble Grounded Context
        context_str = self.format_context(retrieved_chunks)
        user_prompt = (
            f"Provided Context:\n"
            f"====================\n"
            f"{context_str}\n"
            f"====================\n\n"
            f"User Question: {user_query}\n\n"
            f"Answer the question strictly based on the context above. Include [Source: <filename>, Chunk: <chunk_id>] citations."
        )

        # Step 3: Invoke Gemini
        t_gen_start = time.perf_counter()
        try:
            response = self.client.models.generateContent(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=RAG_SYSTEM_INSTRUCTION,
                    temperature=temperature,
                    top_p=0.9,
                ) if types else None,
            )
            raw_answer = response.text or ""
        except Exception as e:
            raw_answer = f"Error communicating with Gemini API: {str(e)}"

        t_gen_end = time.perf_counter()
        generation_ms = (t_gen_end - t_gen_start) * 1000

        # Step 4: Verify Grounding and Extract Citations
        is_refusal = (
            NOT_FOUND_PHRASE.lower() in raw_answer.lower()
            or "do not contain information" in raw_answer.lower()
            or "does not contain" in raw_answer.lower()
        )
        grounded = not is_refusal

        # Match cited sources in response
        citations: List[Dict[str, Any]] = []
        cited_pattern = re.findall(r"\[Source:\s*([^,\]]+),\s*Chunk:\s*([^\]]+)\]", raw_answer)
        seen_cites = set()

        for src, cid in cited_pattern:
            src = src.strip()
            cid = cid.strip()
            key = (src, cid)
            if key not in seen_cites:
                seen_cites.add(key)
                # Find matching chunk snippet
                matching_chunk = next(
                    (c for c in retrieved_chunks if c.get("chunk_id") == cid or c.get("source") == src),
                    None,
                )
                citations.append({
                    "source": src,
                    "chunk_id": cid,
                    "similarity_score": matching_chunk.get("similarity_score", 0.0) if matching_chunk else None,
                    "excerpt": (matching_chunk.get("text", "")[:180] + "...") if matching_chunk else "",
                })

        # If model didn't cite with exact pattern but grounded chunks exist, attach top retrieved chunk
        if grounded and not citations and retrieved_chunks:
            for top_c in retrieved_chunks[:2]:
                citations.append({
                    "source": top_c["source"],
                    "chunk_id": top_c["chunk_id"],
                    "similarity_score": top_c["similarity_score"],
                    "excerpt": top_c["text"][:180] + "...",
                })

        return RagResponse(
            query=user_query,
            answer=raw_answer,
            grounded=grounded,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            model=self.model_name,
        )
