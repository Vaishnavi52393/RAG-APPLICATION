"""
ingest.py - Document Ingestion, Parsing, and Recursive Text Chunking.
Part of the Cloud-Based RAG Document Assistant.

Supports PDF (pypdf), DOCX (python-docx), and plain text/Markdown formats.
Preserves granular document metadata (source name, chunk ID, page number, char count).
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-party document parsers
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore

try:
    import docx
except ImportError:
    docx = None  # type: ignore


@dataclass
class DocumentChunk:
    """Represents an extracted and chunked piece of text with associated metadata."""
    chunk_id: str
    text: str
    source: str
    page_number: Optional[int] = None
    section_index: int = 0
    char_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "page_number": self.page_number if self.page_number is not None else 1,
            "section_index": self.section_index,
            "char_count": self.char_count,
            **self.metadata,
        }


def compute_doc_hash(content: Union[str, bytes]) -> str:
    """Generate SHA-256 fingerprint for document content deduplication."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def extract_text_from_pdf(file_input: Union[bytes, io.BytesIO, Path, str]) -> List[tuple[int, str]]:
    """
    Extract text per page from a PDF file.
    Returns a list of (page_number, text) tuples (1-indexed).
    """
    if PdfReader is None:
        raise ImportError("pypdf is required for PDF parsing. Run 'pip install pypdf'.")

    stream: Union[io.BytesIO, str, Path]
    if isinstance(file_input, bytes):
        stream = io.BytesIO(file_input)
    else:
        stream = file_input

    reader = PdfReader(stream)
    pages: List[tuple[int, str]] = []

    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            # Clean non-printable characters while preserving structure
            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", page_text).strip()
            if cleaned:
                pages.append((idx, cleaned))
        except Exception as e:
            # Continue extracting remaining pages if one has rendering quirks
            pages.append((idx, f"[Warning: Text extraction failed on page {idx}: {e}]"))

    if not pages:
        return [(1, "[Empty PDF document or non-extractable scanned raster]")]
    return pages


def extract_text_from_docx(file_input: Union[bytes, io.BytesIO, Path, str]) -> List[tuple[int, str]]:
    """
    Extract text paragraphs and tables from a DOCX file.
    Returns a list of (section_or_page, text) tuples.
    """
    if docx is None:
        raise ImportError("python-docx is required for DOCX parsing. Run 'pip install python-docx'.")

    stream: Union[io.BytesIO, str, Path]
    if isinstance(file_input, bytes):
        stream = io.BytesIO(file_input)
    else:
        stream = file_input

    doc = docx.Document(stream)
    paragraphs: List[str] = []

    for para in doc.paragraphs:
        txt = para.text.strip()
        if txt:
            paragraphs.append(txt)

    # Extract tables into readable key-value / rows format
    for table in doc.tables:
        table_rows = []
        for row in table.rows:
            row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(row_cells):
                table_rows.append(" | ".join(row_cells))
        if table_rows:
            paragraphs.append("\n[Table Data]\n" + "\n".join(table_rows))

    full_text = "\n\n".join(paragraphs).strip()
    return [(1, full_text if full_text else "[Empty DOCX document]")]


def extract_text_from_txt(file_input: Union[bytes, str, Path]) -> List[tuple[int, str]]:
    """
    Extract text from plain UTF-8 text or Markdown.
    Supports auto-fallback to latin-1 / cp1252 if UTF-8 decode fails.
    """
    if isinstance(file_input, bytes):
        try:
            raw = file_input.decode("utf-8")
        except UnicodeDecodeError:
            raw = file_input.decode("latin-1", errors="replace")
    elif isinstance(file_input, (str, Path)):
        p = Path(file_input)
        try:
            raw = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = p.read_text(encoding="latin-1", errors="replace")
    else:
        raw = str(file_input)

    cleaned = raw.replace("\r\n", "\n").strip()
    return [(1, cleaned if cleaned else "[Empty text document]")]


def recursive_character_chunker(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Splits text hierarchically using semantic boundaries:
    1. Double newlines (paragraphs)
    2. Single newlines (lines)
    3. Sentence periods / punctuation
    4. Spaces (words)
    5. Characters fallback
    Ensures chunks do not exceed `chunk_size` while maintaining `chunk_overlap`.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")

    if not text or not text.strip():
        return []

    if separators is None:
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    def _split_text(content: str, current_seps: List[str]) -> List[str]:
        if len(content) <= chunk_size:
            return [content.strip()] if content.strip() else []

        if not current_seps:
            # Fallback hard slice if no separator matches
            slices = []
            start = 0
            step = chunk_size - chunk_overlap
            while start < len(content):
                slices.append(content[start : start + chunk_size].strip())
                start += step
            return [s for s in slices if s]

        sep = current_seps[0]
        remaining_seps = current_seps[1:]

        # Split on the highest-priority separator
        if sep:
            splits = content.split(sep)
        else:
            splits = list(content)

        accumulated_chunks: List[str] = []
        current_chunk = ""

        for part in splits:
            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    accumulated_chunks.append(current_chunk.strip())
                # If a single part exceeds chunk size, recurse deeper
                if len(part) > chunk_size:
                    deeper = _split_text(part, remaining_seps)
                    accumulated_chunks.extend(deeper)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk and current_chunk.strip():
            accumulated_chunks.append(current_chunk.strip())

        # Now merge or apply sliding window overlap if needed
        final_chunks: List[str] = []
        for i, chk in enumerate(accumulated_chunks):
            if not chk:
                continue
            if not final_chunks:
                final_chunks.append(chk)
            else:
                # Add overlap context from preceding chunk if not already present
                if chunk_overlap > 0 and len(chk) + chunk_overlap <= chunk_size:
                    prev_tail = final_chunks[-1][-chunk_overlap:].strip()
                    if prev_tail and not chk.startswith(prev_tail):
                        chk = f"{prev_tail} ... {chk}"
                final_chunks.append(chk)

        return final_chunks

    raw_chunks = _split_text(text, separators)
    # Deduplicate and strip
    return [c.strip() for c in raw_chunks if len(c.strip()) > 10]


def process_document(
    filename: str,
    file_bytes_or_text: Union[bytes, str],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    """
    High-level ingestion pipeline that auto-detects file extension,
    extracts structured text per page/section, and returns tagged DocumentChunks.
    """
    ext = Path(filename).suffix.lower()
    doc_hash = compute_doc_hash(file_bytes_or_text)

    # 1. Parse text by format
    if ext == ".pdf":
        if isinstance(file_bytes_or_text, str):
            pages = extract_text_from_pdf(file_bytes_or_text.encode("utf-8"))
        else:
            pages = extract_text_from_pdf(file_bytes_or_text)
    elif ext in [".docx", ".doc"]:
        if isinstance(file_bytes_or_text, str):
            pages = extract_text_from_docx(file_bytes_or_text.encode("utf-8"))
        else:
            pages = extract_text_from_docx(file_bytes_or_text)
    elif ext in [".txt", ".md", ".markdown", ".csv", ".json", ".log"]:
        pages = extract_text_from_txt(file_bytes_or_text)
    else:
        # Fallback to plain text decoding
        pages = extract_text_from_txt(file_bytes_or_text)

    chunks: List[DocumentChunk] = []
    chunk_counter = 1

    # 2. Chunk each page/section
    for page_num, page_content in pages:
        if not page_content.strip():
            continue

        raw_chunks = recursive_character_chunker(
            page_content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for sec_idx, chunk_text in enumerate(raw_chunks):
            cid = f"{Path(filename).stem}_c{chunk_counter:03d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    text=chunk_text,
                    source=filename,
                    page_number=page_num,
                    section_index=sec_idx,
                    char_count=len(chunk_text),
                    metadata={
                        "doc_hash": doc_hash,
                        "file_type": ext.lstrip("."),
                    },
                )
            )
            chunk_counter += 1

    return chunks
