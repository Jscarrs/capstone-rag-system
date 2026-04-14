"""
Ingest Single File — Strict Adobe-Only Pipeline

Loads a document (text or PDF), splits into LangChain Document chunks,
and stores in ChromaDB. For PDFs, delegates ALL extraction to adobe_ocr.py.

Chunk Overlap is set to 300 to ensure figure references like "see Figure 2"
are captured alongside descriptive text.

Noise filtering: text chunks with fewer than MIN_CHUNK_ALNUM_CHARS (default 10)
alphanumeric characters are skipped (e.g. stray symbols, footnote markers).

Metadata Schema for every chunk:
  - source: filename
  - path: absolute file path
  - chunk_type: 'text' | 'table' | 'figure'
  - page: page number (1-indexed)
  - figure_id: (figures only) e.g. "Figure[1]"
  - image_path: (figures and tables) absolute path to .png rendition
  - bounds_x_min, bounds_y_min, bounds_x_max, bounds_y_max: PDF coordinate bounds
    (Adobe coordinate system: origin at bottom-left, Y increases upward)
"""

import os
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from shared import get_embeddings, split_text, SCRIPT_DIR, CHROMA_DIR

MIN_ALNUM_CHARS = int(os.getenv("MIN_CHUNK_ALNUM_CHARS", "10"))


def _is_noise_chunk(text):
    """Return True if text is too short or has too few alphanumeric chars to be meaningful."""
    alnum_count = len(re.findall(r'[a-zA-Z0-9]', text))
    return alnum_count < MIN_ALNUM_CHARS


def _extract_bounds_metadata(chunk_data):
    """Extract bounding box fields from chunk data for ChromaDB storage."""
    bounds = chunk_data.get("bounds", [])
    if bounds and len(bounds) >= 4:
        return {
            "bounds_x_min": float(bounds[0]),
            "bounds_y_min": float(bounds[1]),
            "bounds_x_max": float(bounds[2]),
            "bounds_y_max": float(bounds[3]),
        }
    return {}


def prepare_documents(file_path, chunk_size=1000, chunk_overlap=300):
    """
    Load a document and split into LangChain Document chunks.
    Does NOT write to ChromaDB — returns documents for the caller.
    """
    from pdf_processor import extract_text_from_pdf, is_pdf_file

    print(f"Loading document from {file_path}...")

    documents = []
    file_name = os.path.basename(file_path)

    print(f"\nProcessing document (chunk_size={chunk_size}, overlap={chunk_overlap})...")

    if is_pdf_file(file_path):
        # ── Adobe-only structured extraction ──
        structured_chunks = extract_text_from_pdf(file_path)

        for chunk_data in structured_chunks:
            chunk_type = chunk_data.get("type", "text")
            page_num = chunk_data.get("page", 1)
            content = chunk_data.get("content", "")
            section_heading = chunk_data.get("section_heading", "")

            # ── ZERO-NULL POLICY ──
            if not content or not content.strip():
                if chunk_type == "figure":
                    fig_id = chunk_data.get("figure_id", f"Figure_{page_num}")
                    content = f"Visual element {fig_id} on page {page_num}"
                    print(f"  Warning: Using fallback content for {fig_id}")
                elif chunk_type == "table":
                    content = f"Table on page {page_num} (content unavailable)"
                    print(f"  Warning: Using fallback for table on page {page_num}")
                else:
                    print(f"  Warning: Skipping empty {chunk_type} on page {page_num}")
                    continue

            # ── TEXT: split into sub-chunks ──
            if chunk_type == "text":
                bounds_meta = _extract_bounds_metadata(chunk_data)
                text_splits = split_text(content, chunk_size, chunk_overlap)
                for i, split_dict in enumerate(text_splits):
                    split_content = split_dict["text"].strip()
                    if not split_content:
                        continue
                    if _is_noise_chunk(split_content):
                        print(f"  Skipping noise chunk on p.{page_num}: {split_content[:40]!r}")
                        continue
                    lines_before = content[: split_dict["start_pos"]].count("\n")
                    meta = {
                        "source": file_name,
                        "path": file_path,
                        "chunk": i,
                        "page": page_num,
                        "chunk_type": "text",
                        "section_heading": section_heading,
                        "start_line": lines_before + 1,
                        **bounds_meta,
                    }
                    documents.append(
                        Document(page_content=split_content, metadata=meta)
                    )

            # ── TABLE: single chunk, Vision-described content + image_path ──
            elif chunk_type == "table":
                bounds_meta = _extract_bounds_metadata(chunk_data)
                image_path = chunk_data.get("image_path")
                meta = {
                    "source": file_name,
                    "path": file_path,
                    "chunk": 0,
                    "page": page_num,
                    "chunk_type": "table",
                    "section_heading": section_heading,
                    **bounds_meta,
                }
                if image_path:
                    meta["image_path"] = image_path
                documents.append(
                    Document(page_content=content.strip(), metadata=meta)
                )

            # ── FIGURE: single chunk, hybrid content + image_path ──
            elif chunk_type == "figure":
                fig_id = chunk_data.get("figure_id", f"Figure_{page_num}")
                image_path = chunk_data.get("image_path")  # from renditions
                bounds_meta = _extract_bounds_metadata(chunk_data)
                meta = {
                    "source": file_name,
                    "path": file_path,
                    "chunk": 0,
                    "page": page_num,
                    "chunk_type": "figure",
                    "section_heading": section_heading,
                    "figure_id": fig_id,
                    "image_path": image_path,
                    **bounds_meta,
                }
                documents.append(
                    Document(page_content=content.strip(), metadata=meta)
                )

        total_chars = sum(
            len(c.get("content", "")) for c in structured_chunks
        )

    else:
        # ── Plain text files ──
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_text(text, chunk_size, chunk_overlap)
        for i, chunk_dict in enumerate(chunks):
            chunk_content = chunk_dict["text"].strip()
            if not chunk_content:
                continue
            lines_before = text[: chunk_dict["start_pos"]].count("\n")
            documents.append(
                Document(
                    page_content=chunk_content,
                    metadata={
                        "source": file_name,
                        "path": file_path,
                        "chunk": i,
                        "chunk_type": "text",
                        "start_line": lines_before + 1,
                    },
                )
            )

        total_chars = len(text)

    # ── Summary ──
    print("Loaded document")
    print(f"Total characters: {total_chars}")
    print(f"Created {len(documents)} chunks")

    txt = sum(1 for d in documents if d.metadata.get("chunk_type") == "text")
    tbl = sum(1 for d in documents if d.metadata.get("chunk_type") == "table")
    fig = sum(1 for d in documents if d.metadata.get("chunk_type") == "figure")

    if tbl > 0 or fig > 0:
        print(f"  - Text chunks: {txt}")
        print(f"  - Table chunks: {tbl}")
        print(f"  - Figure chunks (hybrid with context): {fig}")

    return documents


def ingest_document(file_path, chunk_size=1000, chunk_overlap=300):
    """Load, chunk, embed, and store in ChromaDB."""
    documents = prepare_documents(file_path, chunk_size, chunk_overlap)

    print("\nCreating embeddings...")
    embeddings = get_embeddings()

    print("Storing in ChromaDB vector database...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"\n✓ Successfully ingested document!")
    print(f"✓ Vector database saved to {CHROMA_DIR}")
    print(f"✓ Total chunks stored: {len(documents)}")

    return vectordb


if __name__ == "__main__":
    file_path = os.path.join(SCRIPT_DIR, "data", "book.txt")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print(f"Please place your document at {os.path.join(SCRIPT_DIR, 'data')}")
        print("Supported formats: .txt, .pdf")
    else:
        ingest_document(file_path)
