# RAG Powered Chatbot for Academic Research

A multimodal RAG chatbot for academic PDFs. Ask questions about your documents — including **figures and tables** — with hybrid retrieval, cross-encoder re-ranking, and on-demand Gemini Vision analysis. Built with a React + Vite frontend and a Flask API backend.

## Key Features

- **Hybrid Retrieval**: Ensemble of semantic (HuggingFace sentence-transformers) + BM25 keyword search
- **Cross-Encoder Re-ranking**: `ms-marco-MiniLM-L-6-v2` re-scores candidate chunks for more accurate relevance ranking
- **Agentic Query Rewriting**: LLM rewrites user questions into 1–3 optimized keyword search queries before retrieval, resolving pronouns from chat history
- **Citation Demotion**: Bibliography / reference chunks are deprioritized so actual content sections rank higher
- **PDF Reference Highlighting**: Embedded PDF viewer with bounding-box overlays that show exactly where each cited source appears in the original document
- **Lazy Vision Architecture**: Figures analyzed on-demand at query time via Gemini Vision; result is cached to avoid redundant API calls
- **Pydantic-Validated Figures**: Quality scoring and false-positive filtering for figure elements during ingestion
- **Zero-API Ingestion**: Spatial synthesis builds figure descriptions from bounding-box text — no API calls required during ingestion by default
- **Adobe PDF Extract API**: Sole engine for PDF OCR and structural analysis (text, tables, figure renditions)
- **Multi-provider LLM**: LM Studio (local/free), OpenAI, or Google Gemini — auto-detected from `.env`
- **Local Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (free, no API key)

## Supported LLM Providers

| Provider      | Type  | Cost             | API Key Required |
| ------------- | ----- | ---------------- | ---------------- |
| **LM Studio** | Local | Free             | No               |
| OpenAI        | Cloud | Paid             | Yes              |
| Google Gemini | Cloud | Free tier / Paid | Yes              |

Provider priority (auto-detected): **LM Studio → OpenAI → Google Gemini**

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Step 2: Configure Environment

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
```

Edit `.env` with your credentials:

```bash
# LLM Provider — uncomment the one you want to use
LMSTUDIO_BASE_URL=http://localhost:1234/v1   # Local (free)
# OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-google-key-here

# Adobe PDF Services (REQUIRED for PDF processing)
PDF_SERVICES_CLIENT_ID=your_client_id
PDF_SERVICES_CLIENT_SECRET=your_client_secret

# CORS allow-list for the frontend
FRONTEND_ORIGIN=http://localhost:5173

# Vision — required for multimodal figure analysis
VISION_MODEL_NAME=gemini-3.1-flash-lite-preview
```

Edit `frontend/.env`:

```bash
VITE_API_BASE_URL=http://localhost:8080
```

### Step 3: Run

```bash
# Terminal 1 — backend API
cd rag_system
python3 rag_chatbot.py

# Terminal 2 — frontend UI
cd frontend
npm run dev
```

Open <http://localhost:5173> in your browser.  
The Flask backend runs on <http://localhost:8080>.

---

## How Multimodal RAG Works

### Ingestion Phase

```
PDF Document
    |
    v
+-------------------------------------------------+
|        Adobe PDF Extract API (Cloud)            |
|  Extracts: Text, Tables, Figure Renditions      |
+-------------------------------------------------+
    |           |              |
    v           v              v
  Text       Tables         Figures (.png)
  chunks     (Markdown)         |
                                v
               +-------------------------------+
               | Pydantic Validation           |
               | (area/dimension thresholds,   |
               |  quality scoring 0.0–1.0)     |
               +-------------------------------+
                                |
                                v
               +-------------------------------+
               | Spatial Synthesis (default)   |
               | Scans text inside bounding    |
               | box, sorts top-to-bottom,     |
               | combines caption + context    |
               | ** Zero API calls **          |
               +-------------------------------+
    |           |              |
    +-----------|------------- +
                v
+-------------------------------------------------+
|   HuggingFace Embeddings --> ChromaDB           |
|   Metadata: chunk_type, page, bounds,           |
|             image_path, figure_type,            |
|             quality_score, section_heading      |
+-------------------------------------------------+
```

### Query Phase

```
User Question
    |
    v
+------------------------------------------+
|  LLM Query Rewriting (1–3 keyword        |
|  queries, pronoun resolution from        |
|  chat history)                           |
+------------------------------------------+
    |
    v (run each query)
+------------------------------------------+
|  Hybrid Retriever                        |
|  60% semantic (HuggingFace) +            |
|  40% keyword (BM25)                      |
+------------------------------------------+
    |
    v (merge + deduplicate)
+------------------------------------------+
|  Cross-Encoder Re-ranking                |
|  (ms-marco-MiniLM-L-6-v2)               |
+------------------------------------------+
    |
    v (partition by tier)
+------------------------------------------+
|  Priority: figures/tables > content >   |
|  citations (bibliography demoted)        |
+------------------------------------------+
    |
    +-- Text/Table chunks --> Standard LLM prompt
    |
    +-- Figure chunks with image_path?
            |
            v
    +-------------------------------+
    |  Load .png from disk          |
    |  Base64-encode                |
    |  Send to Gemini Vision API    |
    |  with user's question         |
    |  (cached per figure+question) |
    +-------------------------------+
            |
            v
       Multimodal Response
```

### Chunk Types in ChromaDB

| Type | Content | Key Metadata |
|------|---------|--------------|
| `text` | Plain text passage | `source`, `page`, `chunk_type`, `bounds_*`, `section_heading` |
| `table` | Markdown table | `source`, `page`, `chunk_type`, `bounds_*`, `image_path` |
| `figure` | Spatial synthesis + caption | `source`, `page`, `chunk_type`, `figure_id`, `image_path`, `figure_type`, `quality_score`, `bounds_*` |

---

## PDF Processing (Adobe-Only)

**Adobe PDF Extract API** is the sole engine for PDF processing. No other PDF libraries are used.

### Setup

1. Get credentials from [Adobe Developer Console](https://developer.adobe.com/console)
2. Add to `.env`:
   ```
   PDF_SERVICES_CLIENT_ID=your_client_id
   PDF_SERVICES_CLIENT_SECRET=your_client_secret
   ```
3. `pdfservices-sdk` is included in `requirements.txt`

### What Adobe Extracts

- **Text**: Structured paragraphs and headings with page number and bounding box coordinates
- **Tables**: Structural JSON converted to clean Markdown, with `.png` rendition
- **Figure Renditions**: Physical `.png` images saved to `rag_system/assets/figures/`

---

## Figure Processing Pipeline

### Pydantic Validation (`figure_models.py`)

Every figure element passes through `assess_figure_quality()`:
- Rejects elements with area < 2000 pts (icons, decorative elements)
- Rejects elements with width or height < 30 pts (table borders, lines)
- Scores figures 0.0–1.0 based on caption, context, image size, and bounds
- Classifies type: `diagram`, `chart`, `photograph`, `illustration`, etc.

### Spatial Synthesis (Default — Zero API Calls)

`_synthesize_text_description()` generates rich text from figure layout:
1. Scans all Adobe elements for text **inside** the figure's bounding box
2. Sorts text snippets top-to-bottom by Y-coordinate (visual reading order)
3. Combines: `[Figure Type] + [Caption] + [Internal Text Flow] + [Context]`

### Ingestion-Time Vision (Optional)

Set `ENABLE_VISION_INGESTION=true` in `.env` to also call Gemini Vision during ingestion for richer pre-computed descriptions. Off by default to avoid 429 rate-limit errors.

### Lazy Vision at Query Time

When a retrieved chunk has an `image_path`:
1. ChromaDB returns the figure chunk with `image_path`
2. Image loaded from disk and Base64-encoded
3. Sent to Gemini Vision with the user's specific question
4. Response cached (keyed by image paths + question hash) to avoid duplicate calls
5. Auto-falls back from `VISION_MODEL_NAME` to `gemini-1.5-flash` on 429 errors

---

## PDF Reference Highlighting

When the RAG cites sources, you can see exactly where each passage appears in the original PDF.

### How It Works

1. **Bounding boxes preserved**: Adobe PDF Extract provides spatial coordinates for every element. These are stored as `bounds_x_min`, `bounds_y_min`, `bounds_x_max`, `bounds_y_max` in ChromaDB metadata.
2. **Embedded PDF viewer**: Click any citation `[1]`, `[2]` in a response, or click a PDF in the sidebar, to open the built-in viewer.
3. **Color-coded highlights**: Retrieved sources are shown as semi-transparent overlays on the PDF page. Each citation gets a unique color.
4. **Focus on click**: All highlights appear dimly by default. Clicking a citation prominently focuses that source region.
5. **Quick navigation**: Source page buttons in the toolbar let you jump between pages containing cited content.

### Coordinate System

Adobe uses PDF points with origin at **bottom-left**, Y increasing upward.  
PDF.js renders with origin at **top-left**, Y increasing downward.  
`PdfViewer.jsx` transforms coordinates on render:
```
css_x = bounds_x_min × scale
css_y = (page_height − bounds_y_max) × scale
```

### Requirements

- Documents must be re-ingested to store bounding box metadata.
- Only PDF documents support highlighting (plain text files have no spatial coordinates).

---

## Re-ingestion

After pipeline changes (or after upgrading to store new metadata fields), delete and rebuild:

```bash
rm -rf rag_system/chroma_db
cd rag_system
python3 ingest.py              # All files in data/
# OR
python3 ingest_single_file.py  # Single file (default: data/book.txt)
```

---

## Project Structure

```
capstone-rag-system/
├── .env.example              # Root environment template
├── .gitignore
├── requirements.txt
├── docs/
│   └── plans/                # Design documents
├── frontend/
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js        # Vite config + PDF.js cMaps copy
│   └── src/
│       ├── App.jsx           # Main React UI (chat, upload, document list)
│       ├── PdfViewer.jsx     # Embedded PDF viewer with highlight overlays
│       ├── apiClient.js      # Centralized API client (VITE_API_BASE_URL)
│       ├── main.jsx          # React entry point
│       └── styles.css        # All frontend styling
└── rag_system/
    ├── rag_chatbot.py        # Flask API server + RAG pipeline + lazy Vision
    ├── adobe_ocr.py          # Adobe PDF Services integration
    ├── figure_models.py      # Pydantic validation, quality scoring, spatial synthesis
    ├── ingest.py             # Multi-file ingestion (walks data/ directory)
    ├── ingest_single_file.py # Single-file ingestion (also used by upload endpoint)
    ├── pdf_processor.py      # PDF routing (delegates all extraction to Adobe)
    ├── shared.py             # LLM/embeddings provider selection, split_text, paths
    ├── data/                 # Place documents here (.txt / .pdf)
    ├── assets/
    │   └── figures/          # Extracted figure images (auto-created)
    └── chroma_db/            # Vector database (auto-created)
```

---

## Configuration Reference

### Backend (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_SERVER_HOST` | `0.0.0.0` | Server host |
| `RAG_SERVER_PORT` | `8080` | Server port |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | Allowed CORS origin(s), comma-separated |
| `DEBUG_CHUNKS` | `true` | Print retrieved chunks to console |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum similarity score for retrieval (0.0–1.0) |
| `RETRIEVAL_K` | `3` | Number of chunks to retrieve per query |
| `ENABLE_QUERY_REWRITE` | `true` | LLM rewrites questions into optimized search queries |
| `ENABLE_RERANKING` | `true` | Cross-encoder re-ranks candidates for accuracy |
| `ENABLE_VISION_INGESTION` | `false` | Call Gemini Vision during PDF ingestion |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL (activates local LLM + local embeddings) |
| `USE_LOCAL_EMBEDDINGS` | `false` | Set to `true` to force local HuggingFace embeddings when not using LM Studio |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL_NAME` | `gpt-3.5-turbo` | OpenAI chat model |
| `GOOGLE_API_KEY` | — | Google Gemini API key (for LLM + Vision) |
| `GEMINI_MODEL_NAME` | `gemini-3.1-flash-lite-preview` | Gemini model for chat generation |
| `VISION_MODEL_NAME` | `gemini-3.1-flash-lite-preview` | Gemini model for Vision API calls |
| `PDF_SERVICES_CLIENT_ID` | — | Adobe PDF Services client ID (**required** for PDFs) |
| `PDF_SERVICES_CLIENT_SECRET` | — | Adobe PDF Services client secret (**required** for PDFs) |
| `MIN_CHUNK_ALNUM_CHARS` | `10` | Minimum alphanumeric characters for a valid text chunk |

### Frontend (`frontend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | `http://localhost:8080` | Backend API base URL |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API root / status |
| `/api/health` | GET | Health check |
| `/api/chat` | POST | Send message, receive answer + sources |
| `/api/upload` | POST | Upload and ingest a document (.txt or .pdf) |
| `/api/documents` | GET | List ingested documents |
| `/api/documents/<name>/file` | GET | Serve original document file (for PDF viewer) |
| `/api/figures/<filename>` | GET | Serve extracted figure image |
| `/api/reset-db` | POST | Clear vector database and data directory |
| `/api/clear` | POST | Clear conversation history for a session |
| `/api/debug/search` | GET | Raw ChromaDB similarity search with scores (debug) |

### Chat Request / Response

**POST `/api/chat`**
```json
{ "message": "What does Figure 3 show?", "session_id": "session_abc123" }
```
```json
{
  "answer": "Figure 3 shows ... [1]",
  "sources": [
    {
      "id": 1,
      "source": "paper.pdf",
      "page": 5,
      "chunk_type": "figure",
      "figure_id": "Figure[3]",
      "image_url": "/api/figures/figure_3.png",
      "bounds": { "x_min": 72.0, "y_min": 400.0, "x_max": 540.0, "y_max": 520.0 },
      "preview": "...",
      "reference": "p.5, figure: Figure[3]"
    }
  ],
  "chunks_retrieved": 3
}
```

---

## Credits & Acknowledgments

### Core Frameworks
- **[LangChain](https://github.com/langchain-ai/langchain)** — LLM application framework
- **[ChromaDB](https://github.com/chroma-core/chroma)** — Vector database

### PDF Processing
- **[Adobe PDF Services](https://developer.adobe.com/document-services/)** — Sole PDF OCR / extraction engine

### Embeddings & Re-ranking
- **[HuggingFace sentence-transformers](https://github.com/UKPLab/sentence-transformers)** — Local embeddings (`all-MiniLM-L6-v2`) and cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`)

### LLM Providers
- **[LM Studio](https://lmstudio.ai/)** — Local LLM inference server
- **[OpenAI](https://openai.com/)** — GPT models API
- **[Google Gemini](https://ai.google.dev/)** — Gemini models API (LLM + Vision)

### Web
- **[Flask](https://github.com/pallets/flask)** — Python web framework
- **[React](https://react.dev/)** + **[Vite](https://vite.dev/)** — Frontend
- **[react-pdf](https://github.com/wojtekmaj/react-pdf)** — PDF.js wrapper for the embedded viewer
