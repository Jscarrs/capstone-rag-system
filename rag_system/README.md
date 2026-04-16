# RAG System — Multimodal Document Question Answering

Backend for the Academic Research RAG system. Handles PDF ingestion, hybrid retrieval, cross-encoder re-ranking, and multimodal response generation via Flask API.

## Architecture

```
+-------------------------------------------------------+
|                  INGESTION PHASE                      |
+-------------------------------------------------------+
|  PDF --> Adobe PDF Extract API                        |
|              |-- Text chunks (structured paragraphs)  |
|              |-- Tables (Markdown + .png rendition)   |
|              +-- Figures (.png + Pydantic filter)     |
|                     |                                 |
|                     v                                 |
|  Spatial Synthesis: bounding-box text scan ->         |
|  rich description (zero API calls)                    |
|                     |                                 |
|  Text/Table/Figure chunks                             |
|    --> HuggingFace Embeddings --> ChromaDB            |
+-------------------------------------------------------+

+-------------------------------------------------------+
|                    QUERY PHASE                        |
+-------------------------------------------------------+
|  Question --> LLM Query Rewriting (1–3 queries)       |
|     |                                                 |
|     +--> Hybrid Retriever (60% semantic + 40% BM25)   |
|     +--> Cross-Encoder Re-ranking                     |
|     +--> Priority: figures/tables > content >         |
|          citations (bibliography demoted)             |
|                                                       |
|  If figure chunks with images retrieved:              |
|    -> Load .png -> Base64 -> Gemini Vision API        |
|    -> Question-aware multimodal response              |
|    -> Response cached per figure+question hash        |
|                                                       |
|  If text/table chunks only:                           |
|    -> Standard LLM prompt -> Text response            |
+-------------------------------------------------------+
```

---

## Quick Start

```bash
# 1. Configure environment (from project root)
cp ../.env.example ../.env
# Edit .env: add credentials for your LLM provider, Adobe, and Google Vision

cp ../frontend/.env.example ../frontend/.env

# 2. Place documents in data/ (PDF or TXT)

# 3. Ingest documents
python3 ingest.py              # All files in data/
python3 ingest_single_file.py  # Single file (default: data/book.txt)

# 4. Start backend API
python3 rag_chatbot.py         # API at http://localhost:8080
python3 rag_chatbot.py --cli   # CLI mode (no server)

# 5. Start frontend (separate terminal, from project root)
cd ../frontend && npm run dev  # UI at http://localhost:5173
```

---

## Files

| File | Purpose |
|------|---------|
| `rag_chatbot.py` | Flask API server: RAG pipeline, query rewriting, re-ranking, lazy Vision, all `/api/*` routes |
| `adobe_ocr.py` | Adobe PDF Services integration + figure/table rendition extraction |
| `figure_models.py` | Pydantic validation, quality scoring, spatial synthesis for figures |
| `ingest.py` | Batch ingestion — walks the `data/` directory |
| `ingest_single_file.py` | Single-file ingestion; `prepare_documents()` also used by the upload endpoint |
| `pdf_processor.py` | Thin routing layer — delegates all PDF extraction to `adobe_ocr.py` |
| `shared.py` | LLM/embeddings provider selection, `split_text()`, directory constants |
| `data/` | Place `.txt` / `.pdf` documents here |
| `assets/figures/` | Extracted figure images — auto-created by Adobe OCR |
| `chroma_db/` | Vector database — auto-created at ingestion |

---

## Figure Processing Pipeline

### Pydantic Validation (`figure_models.py`)

Every figure element from Adobe passes through `assess_figure_quality()`:
- **Rejects** elements with area < 2000 pts (icons, decorative elements)
- **Rejects** elements with width or height < 30 pts (borders, lines)
- **Scores** 0.0–1.0 based on caption, context, image presence, and bounds
- **Classifies** type: `diagram`, `chart`, `photograph`, `illustration`, etc.

### Spatial Synthesis (Default — Zero API Calls)

`_synthesize_text_description()` builds a rich text description from layout:
1. Scans all Adobe elements for text **inside** the figure's bounding box
2. Sorts snippets top-to-bottom by Y-coordinate (visual reading order)
3. Combines: `[Figure Type] + [Caption] + [Internal Text Flow] + [Context]`

This is the default. No API calls are made during ingestion.

### Ingestion-Time Vision (Optional)

Set `ENABLE_VISION_INGESTION=true` in `.env` to also call Gemini Vision during upload/ingestion for richer pre-computed descriptions. Disabled by default to prevent 429 rate-limit errors.

### Lazy Vision at Query Time

When a retrieved chunk has an `image_path`:
1. ChromaDB returns the figure chunk including `image_path`
2. Image loaded from disk and Base64-encoded
3. Sent to Gemini Vision with the user's specific question
4. Response cached (keyed by a hash of image paths + question) to avoid duplicate calls
5. Auto-falls back from `VISION_MODEL_NAME` to `gemini-1.5-flash` on 429 errors

---

## Retrieval Pipeline Details

### Query Rewriting

When `ENABLE_QUERY_REWRITE=true`, the LLM converts the user's question into 1–3 keyword-based search queries before retrieval. This resolves pronouns (e.g., "it" → specific entity from history) and breaks multi-part questions into focused sub-queries.

### Hybrid Retrieval

Each query runs through an `EnsembleRetriever`:
- **60% semantic**: ChromaDB cosine similarity (HuggingFace embeddings)
- **40% keyword**: BM25 index built at startup from all documents in ChromaDB

Results from each sub-query are merged, deduplicated (by content hash), and ranked by cross-query frequency.

### Cross-Encoder Re-ranking

When `ENABLE_RERANKING=true`, `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores all candidates against the original user question. This is a separate scoring pass that improves relevance beyond vector similarity.

### Result Partitioning

After re-ranking, results are partitioned into tiers:
1. **Priority**: figure / table chunks (always ranked first)
2. **Content**: standard text chunks
3. **Citations**: bibliography/reference chunks (ranked last)

Final count is capped at `RETRIEVAL_K × 2`.

---

## Noise Filtering

Text chunks with fewer than `MIN_CHUNK_ALNUM_CHARS` (default 10) alphanumeric characters are dropped during ingestion. This removes stray symbols, footnote markers, and page-number-only chunks.

---

## Metadata Schema

Every chunk stored in ChromaDB includes:

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Filename |
| `path` | string | Absolute file path |
| `page` | int | Page number (1-indexed) |
| `chunk_type` | string | `text`, `table`, or `figure` |
| `section_heading` | string | Nearest section heading (if detected) |
| `start_line` | int | Approximate line number in source (text chunks) |
| `bounds_x_min` | float | Adobe PDF coordinate — left edge |
| `bounds_y_min` | float | Adobe PDF coordinate — bottom edge |
| `bounds_x_max` | float | Adobe PDF coordinate — right edge |
| `bounds_y_max` | float | Adobe PDF coordinate — top edge |
| `figure_id` | string | Figure identifier e.g. `Figure[1]` (figures only) |
| `figure_type` | string | `diagram`, `chart`, `photograph`, etc. — stored by `ingest.py` only |
| `quality_score` | float | 0.0–1.0 quality score — stored by `ingest.py` only |
| `image_path` | string | Absolute path to `.png` rendition (figures + tables) |
| `description` | string | Spatial synthesis description — stored by `ingest.py` only |

> **Note on ingestion paths**: `ingest.py` (batch) stores `figure_type`, `quality_score`, and `description` for figure chunks. `ingest_single_file.py` (used by the `/api/upload` endpoint) stores `figure_id` and `image_path` only. Both paths store all `bounds_*`, `source`, `path`, `page`, `chunk_type`, `section_heading`, and `image_path` fields.

> **Note**: Adobe uses PDF point coordinates with origin at **bottom-left**, Y increasing upward. The frontend `PdfViewer.jsx` transforms these to CSS coordinates (top-left origin) when rendering highlight overlays.

---

## Re-ingestion

After any pipeline changes, delete the database and rebuild:

```bash
rm -rf chroma_db
python3 ingest.py
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_SERVER_HOST` | `0.0.0.0` | Server bind host |
| `RAG_SERVER_PORT` | `8080` | Server port |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | Allowed CORS origin(s), comma-separated |
| `DEBUG_CHUNKS` | `true` | Print retrieved chunks to console |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum similarity score for retrieval |
| `RETRIEVAL_K` | `3` | Number of chunks to retrieve per query |
| `ENABLE_QUERY_REWRITE` | `true` | LLM rewrites questions before retrieval |
| `ENABLE_RERANKING` | `true` | Cross-encoder re-ranking of retrieved candidates |
| `ENABLE_VISION_INGESTION` | `false` | Call Gemini Vision during PDF ingestion |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL (activates local LLM + local embeddings) |
| `USE_LOCAL_EMBEDDINGS` | `false` | Set to `true` to force local HuggingFace embeddings when not using LM Studio |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL_NAME` | `gpt-3.5-turbo` | OpenAI chat model |
| `GOOGLE_API_KEY` | — | Google Gemini API key (LLM + Vision) |
| `GEMINI_MODEL_NAME` | `gemini-3.1-flash-lite-preview` | Gemini model for chat generation |
| `VISION_MODEL_NAME` | `gemini-3.1-flash-lite-preview` | Gemini model for Vision API calls |
| `PDF_SERVICES_CLIENT_ID` | — | Adobe PDF Services client ID (**required** for PDFs) |
| `PDF_SERVICES_CLIENT_SECRET` | — | Adobe PDF Services client secret (**required** for PDFs) |
| `MIN_CHUNK_ALNUM_CHARS` | `10` | Noise filter: minimum alphanumeric chars to keep a text chunk |

