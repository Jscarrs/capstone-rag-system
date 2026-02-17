# LangChain Chatbot with Multimodal RAG

A chatbot built with LangChain that supports conversation memory and **Multimodal RAG** (Retrieval-Augmented Generation) — chat with your documents including **figures, tables, and text** with true visual understanding.

## Key Features

- **Lazy Vision Architecture**: Figures analyzed on-demand at query time via Gemini Vision API
- **Pydantic-Validated Figures**: Quality scoring and false-positive filtering for figure elements
- **Zero-API Spatial Synthesis**: Ingestion uses bounding-box text extraction (no API calls, no 429 errors)
- **Strict Adobe PDF Extract API**: Sole engine for PDF OCR and structural analysis
- **High-Fidelity Tables**: Converted to clean Markdown with row/column integrity
- **Web Interface**: Clean, modern chat UI with drag-and-drop document upload
- **Multi-provider LLM**: Supports LM Studio (local/free), OpenAI, and Google Gemini
- **Local Embeddings**: Uses HuggingFace sentence-transformers (free, no API keys)

## Supported LLM Providers

| Provider      | Type  | Cost             | API Key Required |
| ------------- | ----- | ---------------- | ---------------- |
| **LM Studio** | Local | Free             | No               |
| OpenAI        | Cloud | Paid             | Yes              |
| Google Gemini | Cloud | Free tier / Paid | Yes              |

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# LLM Provider (choose one)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
# OPENAI_API_KEY=sk-your-key-here
# GOOGLE_API_KEY=your-google-key-here

# Adobe PDF Services (REQUIRED for PDF processing)
PDF_SERVICES_CLIENT_ID=your_client_id
PDF_SERVICES_CLIENT_SECRET=your_client_secret

# Vision (optional, for lazy figure analysis at query time)
GOOGLE_API_KEY=your-google-key-here
VISION_MODEL_NAME=gemini-2.5-flash
```

### Step 3: Run

```bash
cd rag_system
python3 rag_chatbot.py
```

Open <http://localhost:8080> in your browser.
Upload documents directly through the web UI.

---

## How Multimodal RAG Works

### Ingestion Phase (Zero-API)

```
PDF Document
    |
    v
+-------------------------------------------------+
|       Adobe PDF Extract API (Cloud)             |
|  Extracts: Text, Tables, Figure Renditions      |
+-------------------------------------------------+
    |           |              |
    v           v              v
  Text       Tables         Figures
 chunks     (Markdown)     |
                           v
              +-------------------------------+
              | Pydantic Validation           |
              | (FigureCandidate -> quality   |
              |  scoring -> reject/accept)    |
              +-------------------------------+
                           |
                           v
              +-------------------------------+
              | Spatial Synthesis (default)   |
              | Scans text INSIDE bounding    |
              | box, sorts top-to-bottom,     |
              | combines caption + context    |
              | ** Zero API calls **          |
              +-------------------------------+
    |           |              |
    +-----------|------------- +
                v
+-------------------------------------------------+
|   HuggingFace Embeddings -> ChromaDB            |
|   Metadata: chunk_type, page, image_path,       |
|             figure_type, quality_score           |
+-------------------------------------------------+
```

### Query Phase (Lazy Vision)

```
User Question
    |
    v
+---------------------------------------+
|  Similarity Search -> Top K Chunks    |
+---------------------------------------+
    |
    +-- Text/Table chunks -> Standard LLM prompt
    |
    +-- Figure chunks with image_path?
            |
            v
    +----------------------------------+
    |  Load .png from disk -> Base64   |
    |  Send to Gemini Vision API       |
    |  with user's actual question     |
    |  (question-aware analysis)       |
    +----------------------------------+
            |
            v
       Multimodal Response
       (text + visual analysis)
       Cached for repeat queries
```

### Chunk Types in ChromaDB

| Type | Content | Metadata |
|------|---------|----------|
| `text` | Plain text passage | `source`, `page`, `chunk_type` |
| `table` | Markdown table | `source`, `page`, `chunk_type` |
| `figure` | Spatial description + caption + context | `source`, `page`, `chunk_type`, `figure_id`, `image_path`, `figure_type`, `quality_score` |

---

## PDF Processing (Adobe-Only)

**Adobe PDF Extract API** is the sole engine for PDF processing. No other libraries (PyMuPDF, pdfplumber, pypdf) are used.

### Setup

1. Get credentials from [Adobe Developer Console](https://developer.adobe.com/console)
2. Add to `.env`:
   ```
   PDF_SERVICES_CLIENT_ID=your_client_id
   PDF_SERVICES_CLIENT_SECRET=your_client_secret
   ```
3. SDK is included in `requirements.txt`

### What Adobe Extracts

- **Text**: Structured paragraphs, headings, lists with page/bounds metadata
- **Tables**: Structural JSON converted to clean Markdown
- **Figure Renditions**: Physical `.png` images saved to `rag_system/assets/figures/`

---

## Figure Processing Pipeline

### Pydantic Validation

Every figure element passes through `assess_figure_quality()` which:
- Rejects elements with area < 2000 pts (icons, decorative elements)
- Rejects elements with width or height < 30 pts (table fragments, lines)
- Scores figures 0.0-1.0 based on caption, context, image size, bounds
- Classifies type: `diagram`, `chart`, `photograph`, `illustration`, etc.

### Spatial Synthesis (Default, Zero-API)

For accepted figures, `_synthesize_text_description()`:
1. Scans all Adobe elements for text **inside** the figure's bounding box
2. Sorts internal text top-to-bottom by Y-coordinate (visual reading order)
3. Combines: `[Figure Type] + [Caption] + [Internal Text Flow] + [Context]`

### Ingestion-Time Vision (Optional)

Set `ENABLE_VISION_INGESTION=true` in `.env` to also call Gemini Vision during ingestion for richer pre-computed descriptions. Off by default to avoid 429 rate-limit errors.

### Lazy Vision at Query Time

When a user asks about a figure:
1. ChromaDB retrieves the figure chunk with `image_path`
2. The image is loaded from disk and Base64-encoded
3. Sent to Gemini Vision with the user's specific question
4. Response is cached to avoid duplicate API calls

---

## Re-ingestion

**Important**: If you change the ingestion pipeline, delete and rebuild:

```bash
rm -rf rag_system/chroma_db
cd rag_system
python3 ingest.py              # Multi-file ingestion
# OR
python3 ingest_single_file.py  # Single-file ingestion
```

---

## Project Structure

```
capstone-rag-system/
├── requirements.txt
├── .env.example
├── .gitignore
└── rag_system/
    ├── rag_chatbot.py        # RAG chatbot server + lazy Vision
    ├── adobe_ocr.py          # Adobe PDF Services integration
    ├── figure_models.py      # Pydantic validation + spatial synthesis
    ├── ingest.py             # Multi-file ingestion
    ├── ingest_single_file.py # Single-file ingestion
    ├── pdf_processor.py      # PDF routing (delegates to Adobe)
    ├── data/                 # Place documents here
    ├── assets/
    │   └── figures/          # Extracted figure images (.png)
    ├── static/               # Web frontend
    │   ├── index.html
    │   ├── styles.css
    │   └── app.js
    └── chroma_db/            # Vector database (auto-created)
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_SERVER_HOST` | `0.0.0.0` | Server host |
| `RAG_SERVER_PORT` | `8080` | Server port |
| `DEBUG_CHUNKS` | `true` | Show retrieved chunks in console |
| `SIMILARITY_THRESHOLD` | `0.4` | RAG retrieval threshold (0.0-1.0) |
| `RETRIEVAL_K` | `3` | Number of chunks to retrieve per query |
| `ENABLE_VISION_INGESTION` | `false` | Call Gemini Vision during PDF ingestion |
| `VISION_MODEL_NAME` | `gemini-2.5-flash` | Gemini model for Vision API calls |
| `PDF_SERVICES_CLIENT_ID` | - | Adobe API client ID |
| `PDF_SERVICES_CLIENT_SECRET` | - | Adobe API client secret |
| `GOOGLE_API_KEY` | - | Gemini API key (for Vision + LLM) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/chat` | POST | Send message, get response |
| `/api/upload` | POST | Upload and ingest a document |
| `/api/documents` | GET | List ingested documents |
| `/api/reset-db` | POST | Reset vector database |
| `/api/clear` | POST | Clear conversation history |
| `/api/health` | GET | Health check |

---

## Credits & Acknowledgments

### Core Frameworks

- **[LangChain](https://github.com/langchain-ai/langchain)** — Framework for building LLM-powered applications
- **[ChromaDB](https://github.com/chroma-core/chroma)** — Open-source embedding database

### PDF Processing

- **[Adobe PDF Services](https://developer.adobe.com/document-services/)** — Cloud-based PDF extraction via Extract API (sole OCR engine)

### Embeddings

- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** — Local embeddings via `sentence-transformers/all-MiniLM-L6-v2`

### LLM Providers

- **[LM Studio](https://lmstudio.ai/)** — Local LLM inference server
- **[OpenAI](https://openai.com/)** — GPT models API
- **[Google Gemini](https://ai.google.dev/)** — Gemini models API (including Vision)

### Web Framework

- **[Flask](https://github.com/pallets/flask)** — Python web framework for the chat server
