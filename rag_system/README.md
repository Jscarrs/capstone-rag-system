# RAG System - Document Question Answering

A Retrieval-Augmented Generation (RAG) system that allows you to chat with your documents. Supports both OpenAI and Google Gemini.

## How It Works

1. **Ingestion**: Text is split into chunks, embedded using OpenAI or Gemini embeddings, and stored in a local ChromaDB vector database
2. **Retrieval**: When you ask a question, the system finds the most relevant chunks using similarity search
3. **Generation**: The LLM generates an answer based on the retrieved context

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your API key in `.env` (choose one):
```
OPENAI_API_KEY=sk-your-actual-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

3. Place your text file (book, document, etc.) at:
```
rag_system/data/book.txt
```

4. Ingest the document (create embeddings and vector database):
```bash
python rag_system/ingest.py
```

This will:
- Load your text file
- Split it into chunks (default: 1000 characters with 200 overlap)
- Create embeddings for each chunk (OpenAI or Gemini based on configured key)
- Store in ChromaDB at `./rag_system/chroma_db`

5. Run the RAG chatbot:
```bash
python rag_system/rag_chatbot.py
```

## Important Note

You must use the same embedding provider for both ingestion and querying. If you ingest with OpenAI embeddings, you must query with OpenAI embeddings (and vice versa for Gemini).

## Files

- `ingest.py` - Script to load, chunk, embed, and store documents
- `rag_chatbot.py` - Chatbot that queries the vector database to answer questions
- `data/` - Place your text files here
- `chroma_db/` - Local vector database (created after running ingest.py)

## Customization

In `ingest.py`, you can adjust:
- `chunk_size` - Size of text chunks (default: 1000 characters)
- `chunk_overlap` - Overlap between chunks (default: 200 characters)

In `rag_chatbot.py`, you can adjust:
- `k` in `search_kwargs` - Number of chunks to retrieve (default: 3)
