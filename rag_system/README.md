# RAG System - Document Question Answering

A Retrieval-Augmented Generation (RAG) system that allows you to chat with your documents.

## Supported Providers

| Component | Local (Free) | Cloud |
|-----------|--------------|-------|
| **LLM** | LM Studio | OpenAI, Gemini |
| **Embeddings** | HuggingFace | OpenAI, Gemini |
| **Vector DB** | ChromaDB | ChromaDB |

---

## Quick Start with LM Studio (Recommended)

### 1. Start LM Studio Server

1. Open LM Studio
2. Load a model (e.g., Qwen, Llama, Mistral)
3. Go to **Local Server** tab → **Start Server**

### 2. Configure `.env`

```
LMSTUDIO_BASE_URL=http://localhost:1234/v1
```

### 3. Ingest Documents

```bash
cd rag_system
python3 ingest.py
```

### 4. Chat with Documents

```bash
python3 rag_chatbot.py
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PHASE                         │
├─────────────────────────────────────────────────────────────┤
│  data/book.txt → Split into → HuggingFace → ChromaDB       │
│                  1000-char    Embeddings    Vector DB       │
│                  chunks       (local)       (local)         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PHASE                            │
├─────────────────────────────────────────────────────────────┤
│  Question → Similarity → Top 3 → Inject as → LM Studio     │
│             Search       Chunks   Context     Response      │
└─────────────────────────────────────────────────────────────┘
```

---

## Adding Documents

1. Place `.txt` files in `data/` folder
2. Delete old database: `rm -rf chroma_db`
3. Re-ingest: `python3 ingest.py`

---

## Cloud Providers (Alternative)

Edit `.env` to use cloud APIs instead:

```bash
# Remove or comment out LMSTUDIO_BASE_URL
# LMSTUDIO_BASE_URL=http://localhost:1234/v1

# Use one of these:
OPENAI_API_KEY=sk-your-key-here
# OR
GOOGLE_API_KEY=your-google-key-here
```

**Note**: If switching between local and cloud embeddings, delete `chroma_db/` first (different dimensions).

---

## Files

| File | Purpose |
|------|---------|
| `ingest.py` | Load, chunk, embed, and store documents |
| `rag_chatbot.py` | Query vector DB and generate answers |
| `data/` | Place your `.txt` documents here |
| `chroma_db/` | Vector database (auto-created) |

---

## Customization

**ingest.py**:
- `chunk_size` - Size of text chunks (default: 1000 characters)
- `chunk_overlap` - Overlap between chunks (default: 200 characters)

**rag_chatbot.py**:
- `k` in `search_kwargs` - Number of chunks to retrieve (default: 3)
