# RAG System — Multimodal Document Question Answering

A Retrieval-Augmented Generation system that lets you chat with your documents, including **visual understanding of figures and tables** via hybrid retrieval and on-demand Gemini Vision analysis.

## Architecture

```
+-------------------------------------------------------+
|                  INGESTION PHASE                       |
+-------------------------------------------------------+
|  PDF --> Adobe PDF Extract API                         |
|              |-- Text chunks (structured paragraphs)   |
|              |-- Tables (cell text + rendition .png)   |
|              +-- Figures (-> .png + Pydantic filter)   |
|                     |                                  |
|                     v                                  |
|  Text/Table/Figure chunks                              |
|    --> HuggingFace Embeddings --> ChromaDB              |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                    QUERY PHASE                          |
+-------------------------------------------------------+
|  Question --> Text embedding (HuggingFace)             |
|     |                                                  |
|     +--> Search collection (hybrid: vector+BM25)       |
|     +--> Rank + dedupe results                         |
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
cp ../frontend/.env.example ../frontend/.env

# 2. Ingest documents
python3 ingest.py              # All files in data/
python3 ingest_single_file.py  # Single file

# 3. Start backend API
python3 rag_chatbot.py         # API at http://localhost:8080
python3 rag_chatbot.py --cli   # CLI mode

# 4. Start frontend (run from project root in another terminal)
cd ../frontend
npm install
npm run dev                    # UI at http://localhost:5173
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
5. Model auto-falls back from `VISION_MODEL_NAME` to `gemini-1.5-flash` on 429 errors

---

## Files

| File | Purpose |
|------|---------|
| `rag_chatbot.py` | RAG chatbot server with retrieval + lazy Vision |
| `adobe_ocr.py` | Adobe PDF Services integration + rendition extraction |
| `figure_models.py` | Pydantic validation, quality scoring, spatial synthesis |
| `ingest.py` | Multi-file ingestion to ChromaDB |
| `ingest_single_file.py` | Single-file ingestion to ChromaDB |
| `pdf_processor.py` | PDF routing (delegates to Adobe) |
| `shared.py` | Provider config, embedding selection, shared constants |
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

After pipeline changes, delete both databases and rebuild:

```bash
rm -rf chroma_db
python3 ingest.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum similarity score for retrieval |
| `RETRIEVAL_K` | `3` | Number of chunks to retrieve per query |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | Allowed frontend origin(s) for CORS (comma-separated) |
| `ENABLE_VISION_INGESTION` | `false` | Call Gemini Vision during PDF ingestion |
| `GEMINI_MODEL_NAME` | `gemini-3.1-flash-lite-preview` | Gemini model for chat generation |
| `VISION_MODEL_NAME` | `gemini-3.1-flash-lite-preview` | Gemini model for Vision API calls |
| `chunk_size` | `1000` | Text chunk size in characters (`ingest_single_file.py`) |
| `chunk_overlap` | `300` | Overlap between text chunks (`ingest_single_file.py`) |
