# RAG System — Multimodal Document Question Answering

A Retrieval-Augmented Generation system that lets you chat with your documents, including **visual understanding of figures and tables** via on-demand Gemini Vision analysis.

## Architecture

```
+-------------------------------------------------------+
|                  INGESTION PHASE                       |
+-------------------------------------------------------+
|  PDF --> Adobe PDF Extract API                         |
|              |-- Text chunks (structured paragraphs)   |
|              |-- Tables (-> Markdown with integrity)   |
|              +-- Figures (-> .png + Pydantic filter)   |
|                     |                                  |
|                     v                                  |
|              Pydantic Validation                       |
|              (quality scoring, false-positive filter)  |
|                     |                                  |
|                     v                                  |
|              Spatial Synthesis (zero API calls)        |
|              Scans text inside bounding box,           |
|              sorts top-to-bottom by Y-coordinate       |
|                                                        |
|  All chunks --> HuggingFace Embeddings --> ChromaDB    |
|                 (local, free)              (local)     |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                    QUERY PHASE                          |
+-------------------------------------------------------+
|  Question --> Similarity Search --> Top K Chunks        |
|                                                        |
|  If figure chunks with images retrieved:               |
|    -> Load .png -> Base64 -> Gemini Vision API         |
|    -> Question-aware multimodal response               |
|    -> Response cached for repeat queries               |
|                                                        |
|  If text/table chunks only:                            |
|    -> Standard LLM prompt -> Text response             |
+-------------------------------------------------------+
```

---

## Quick Start

```bash
# 1. Configure credentials
cp ../.env.example ../.env
# Edit .env: add PDF_SERVICES_CLIENT_ID, PDF_SERVICES_CLIENT_SECRET, GOOGLE_API_KEY

# 2. Ingest documents
python3 ingest.py              # All files in data/
python3 ingest_single_file.py  # Single file

# 3. Start server
python3 rag_chatbot.py         # Web UI at http://localhost:8080
python3 rag_chatbot.py --cli   # CLI mode
```

---

## How Figure Processing Works

### Pydantic Validation (`figure_models.py`)

Every figure element from Adobe passes through `assess_figure_quality()`:
- **Rejects** elements with area < 2000 pts (icons, decorative elements)
- **Rejects** elements with width or height < 30 pts (table borders, lines)
- **Scores** 0.0-1.0 based on caption, context, image size, and bounds
- **Classifies** type: `diagram`, `chart`, `photograph`, `illustration`, etc.

### Spatial Synthesis (Default, Zero-API)

`_synthesize_text_description()` generates rich text from figure layout:
1. Scans all Adobe elements for text **inside** the figure's bounding box
2. Sorts snippets top-to-bottom by Y-coordinate (visual reading order)
3. Combines: `[Figure Type] + [Caption] + [Internal Text Flow] + [Context]`

No API calls during ingestion by default. Set `ENABLE_VISION_INGESTION=true` in `.env` to also call Gemini Vision during upload.

### Lazy Vision at Query Time

When a user asks about a figure:
1. ChromaDB retrieves the figure chunk with `image_path`
2. Image loaded from disk, Base64-encoded
3. Sent to Gemini Vision with the user's **specific question**
4. Response cached to avoid duplicate API calls
5. Model auto-falls back from `gemini-2.5-flash` to `gemini-1.5-flash` on 429 errors

---

## Files

| File | Purpose |
|------|---------|
| `rag_chatbot.py` | RAG chatbot server with lazy Vision at query time |
| `adobe_ocr.py` | Adobe PDF Services integration + rendition extraction |
| `figure_models.py` | Pydantic validation, quality scoring, spatial synthesis |
| `ingest.py` | Multi-file ingestion to ChromaDB |
| `ingest_single_file.py` | Single-file ingestion to ChromaDB |
| `pdf_processor.py` | PDF routing (delegates to Adobe) |
| `data/` | Place your `.txt` / `.pdf` documents here |
| `assets/figures/` | Extracted figure images (auto-created) |
| `chroma_db/` | Vector database (auto-created) |

---

## Metadata Schema

Every chunk stored in ChromaDB includes:

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Filename |
| `path` | string | Absolute file path |
| `page` | int | Page number (1-indexed) |
| `chunk_type` | string | `text`, `table`, or `figure` |
| `figure_id` | string | Figure identifier (figures only) |
| `figure_type` | string | `diagram`, `chart`, `photograph`, etc. (figures only) |
| `quality_score` | float | 0.0-1.0 quality score (figures only) |
| `image_path` | string | Path to `.png` rendition (figures only) |
| `description` | string | Spatial synthesis or Vision description (figures only) |

---

## Re-ingestion

After pipeline changes, delete and rebuild:

```bash
rm -rf chroma_db
python3 ingest.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | `0.4` | Minimum similarity score for retrieval |
| `RETRIEVAL_K` | `3` | Number of chunks to retrieve per query |
| `ENABLE_VISION_INGESTION` | `false` | Call Gemini Vision during PDF ingestion |
| `VISION_MODEL_NAME` | `gemini-2.5-flash` | Gemini model for Vision API calls |
| `chunk_size` | `1000` | Text chunk size in characters (`ingest_single_file.py`) |
| `chunk_overlap` | `300` | Overlap between text chunks (`ingest_single_file.py`) |
