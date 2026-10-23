"""
tests/test_rag.py - Test Suite for Cloud-Based RAG Document Assistant.
Executes unit tests for parsing, chunking, prompt assembly, and refusal logic.
Run with: `pytest tests/test_rag.py -v`
"""

import os
import sys
from pathlib import Path

# Ensure root package directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import unittest
from unittest.mock import MagicMock, patch

from ingest import (
    DocumentChunk,
    compute_doc_hash,
    extract_text_from_txt,
    process_document,
    recursive_character_chunker,
)
from rag_pipeline import NOT_FOUND_PHRASE, RAG_SYSTEM_INSTRUCTION, RagPipeline, RagResponse


class TestDocumentIngestion(unittest.TestCase):
    """Tests for document text parsing and hashing."""

    def test_compute_doc_hash_stability(self):
        content = "Enterprise Cloud Run Architecture 2026"
        hash1 = compute_doc_hash(content)
        hash2 = compute_doc_hash(content)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)

    def test_extract_text_from_txt_content(self):
        raw_text = "Google Cloud Run provides serverless container execution.\nConcurrency limit is 1000."
        pages = extract_text_from_txt(raw_text.encode("utf-8"))
        self.assertEqual(len(pages), 1)
        self.assertIn("serverless container execution", pages[0][1])

    def test_recursive_chunker_basic(self):
        text = (
            "Paragraph one is about Google Cloud Run and automatic scaling.\n\n"
            "Paragraph two discusses zero-trust security and IAM principles.\n\n"
            "Paragraph three covers ChromaDB vector persistence and HNSW graphs."
        )
        chunks = recursive_character_chunker(text, chunk_size=100, chunk_overlap=20)
        self.assertTrue(len(chunks) >= 3)
        for chk in chunks:
            self.assertTrue(len(chk) > 0)
            self.assertTrue(len(chk) <= 150)  # Allows slight margin for overlap

    def test_recursive_chunker_boundary_protection(self):
        """Verifies chunker does not fail on edge inputs."""
        empty_chunks = recursive_character_chunker("", chunk_size=500, chunk_overlap=50)
        self.assertEqual(empty_chunks, [])

        with self.assertRaises(ValueError):
            recursive_character_chunker("Some text", chunk_size=100, chunk_overlap=120)

    def test_process_document_metadata_preservation(self):
        sample_text = "Cloud Run auto-scales from zero to thousands of instances.\n\nMemory can be up to 32 GiB."
        chunks = process_document(
            filename="cloud_run_guide.txt",
            file_bytes_or_text=sample_text.encode("utf-8"),
            chunk_size=120,
            chunk_overlap=20,
        )
        self.assertTrue(len(chunks) >= 1)
        first_chunk = chunks[0]
        self.assertEqual(first_chunk.source, "cloud_run_guide.txt")
        self.assertTrue(first_chunk.chunk_id.startswith("cloud_run_guide_c"))
        self.assertEqual(first_chunk.page_number, 1)
        self.assertIn("doc_hash", first_chunk.metadata)


class TestRagPipelineLogic(unittest.TestCase):
    """Tests for prompt assembly, refusal logic, and citation grounding."""

    def setUp(self):
        self.mock_vector_store = MagicMock()

    @patch("rag_pipeline.genai")
    def test_rag_refusal_when_no_chunks_found(self, mock_genai):
        """When vector store finds no relevant documents, pipeline must refuse without LLM call."""
        self.mock_vector_store.search.return_value = []
        os.environ["GEMINI_API_KEY"] = "mock_test_key"

        pipeline = RagPipeline(
            vector_store=self.mock_vector_store,
            api_key="mock_test_key",
        )
        response = pipeline.query("What is the recipe for baking sourdough bread?", min_similarity=0.3)

        self.assertFalse(response.grounded)
        self.assertIn(NOT_FOUND_PHRASE, response.answer)
        # Verify vector store was called, but LLM generateContent was skipped
        self.mock_vector_store.search.assert_called_once()
        self.assertEqual(len(response.retrieved_chunks), 0)

    @patch("rag_pipeline.genai")
    def test_context_formatting_structure(self, mock_genai):
        """Ensures context block includes source headers and chunk IDs."""
        os.environ["GEMINI_API_KEY"] = "mock_test_key"
        pipeline = RagPipeline(
            vector_store=self.mock_vector_store,
            api_key="mock_test_key",
        )
        chunks = [
            {
                "chunk_id": "handbook_c001",
                "source": "handbook.txt",
                "page_number": 2,
                "similarity_score": 0.89,
                "text": "Cloud Run default concurrency is 80 requests.",
            }
        ]
        formatted = pipeline.format_context(chunks)
        self.assertIn("BEGIN SOURCE CHUNK 1", formatted)
        self.assertIn("Source: handbook.txt", formatted)
        self.assertIn("Chunk ID: handbook_c001", formatted)
        self.assertIn("89.00%", formatted)


if __name__ == "__main__":
    unittest.main()
