# LangChain Chatbot with RAG

A chatbot built with LangChain that supports conversation memory and RAG (Retrieval-Augmented Generation) for chatting with your documents.

## Supported LLM Providers

| Provider      | Type  | Cost             | API Key Required |
| ------------- | ----- | ---------------- | ---------------- |
| **LM Studio** | Local | Free             | No               |
| OpenAI        | Cloud | Paid             | Yes              |
| Google Gemini | Cloud | Free tier / Paid | Yes              |

---

## Quick Start with LM Studio (Recommended)

### Step 1: Install LM Studio

Download from https://lmstudio.ai/

### Step 2: Load a Model in LM Studio

1. Open LM Studio
2. Go to **Discover** tab
3. Download a model (recommended: `Qwen2.5-7B-Instruct` or `Llama-3.2-3B-Instruct`)
4. Go to **Local Server** tab
5. Select your model and click **Start Server**
6. Server runs at `http://localhost:1234`

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```
LMSTUDIO_BASE_URL=http://localhost:1234/v1
```

### Step 5: Run

```bash
# Simple chatbot (CLI)
python3 chatbot.py

# RAG chatbot (chat with documents)
cd rag_system
python3 ingest.py      # Ingest documents first
python3 rag_chatbot.py # Start web server (default)
```

Open http://localhost:8080 in your browser to use the web interface.

---

## RAG System (Chat with Documents)

### How It Works

1. **Ingest**: Documents are split into chunks and converted to embeddings
2. **Retrieve**: User questions find relevant chunks via similarity search
3. **Generate**: LLM answers based on retrieved context

### Usage

1. Place your documents (`.txt`) in `rag_system/data/`

2. Ingest documents  
   (this step rebuilds the vector database):

   **Single file ingestion**

   ```bash
   cd rag_system
   python3 ingest_single_file.py
   ```

   **Multiple file ingestion**

   ```bash
   cd rag_system
   python3 ingest.py
   ```

3. Chat with your documents:

   **Web Interface (default):**
   ```bash
   python3 rag_chatbot.py
   ```
   Open http://localhost:8080 in your browser.

   **Command Line Interface:**
   ```bash
   python3 rag_chatbot.py --cli
   ```

### Example Questions

Based on the sample document (Sonic story):

- "What is the Chrono Core?"
- "Who are Sonic's friends?"
- "What happened to Eggman's fortress?"

---

## Cloud Providers (Alternative)

If you prefer cloud APIs, edit `.env`:

```bash
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# OR Google Gemini
GOOGLE_API_KEY=your-google-key-here
```

Remove or comment out `LMSTUDIO_BASE_URL` to use cloud providers.

---

## Project Structure

```
RAG-SYSTEM/
├── chatbot.py              # Simple chatbot (no RAG)
├── requirements.txt
├── .env.example
└── rag_system/
    ├── ingest.py           # Multi-file ingestion
    ├── ingest_single_file.py # Single-file ingestion
    ├── rag_chatbot.py      # RAG chatbot server
    ├── data/               # Place documents here
    │   └── book.txt
    ├── static/             # Web frontend
    │   ├── index.html
    │   ├── styles.css
    │   └── app.js
    └── chroma_db/          # Vector database (auto-created)
```

## Features

- **Web Interface**: Clean, modern chat UI accessible at http://localhost:8080
- **Local-first**: Run 100% locally with LM Studio (free, no API keys)
- **RAG Support**: Chat with your documents with source citations
- **Multi-provider**: Supports LM Studio, OpenAI, and Google Gemini
- **Local Embeddings**: Uses HuggingFace sentence-transformers (free)
- **Conversation Memory**: Maintains chat context per session
- **CLI Mode**: Command-line interface available with `--cli` flag

## Server Configuration

Configure via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_SERVER_HOST` | `0.0.0.0` | Server host |
| `RAG_SERVER_PORT` | `8080` | Server port |
| `DEBUG_CHUNKS` | `true` | Show retrieved chunks in console |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/chat` | POST | Send message, get response |
| `/api/clear` | POST | Clear conversation history |
| `/api/health` | GET | Health check |
