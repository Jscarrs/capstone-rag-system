"""
RAG Chatbot Server

Requirements:
- Flask for web server
- API-only backend (frontend served separately)
- LangChain for RAG pipeline
- ChromaDB for vector storage
- Environment variables for LLM configuration
- Serve extracted figure images via /api/figures/ for frontend display
- Include image_url in source data for figure chunks
- Serve original PDF files via /api/documents/<name>/file for PDF viewer
- Include bounding box coordinates in source data for highlight overlays
- Agentic query rewriting: LLM rewrites user questions into 1-3 optimized
  search queries before retrieval, improving recall for vague/multi-part
  questions and follow-ups that use pronouns
- Citation demotion: bibliography/reference chunks are deprioritized in
  retrieval results so actual content sections rank higher
- Visual retrieval: on-demand Gemini Vision analysis for figure/table
  images, with hybrid text+BM25 retrieval for document search

Environment Variables:
- LMSTUDIO_BASE_URL: Local LM Studio server URL
- OPENAI_API_KEY: OpenAI API key
- GOOGLE_API_KEY: Google Gemini API key
- GEMINI_MODEL_NAME: Gemini model for chat generation
- VISION_MODEL_NAME: Gemini model for vision generation
- USE_LOCAL_EMBEDDINGS: Use local HuggingFace embeddings
- RAG_SERVER_HOST: Server host (default: 0.0.0.0)
- RAG_SERVER_PORT: Server port (default: 8080)
- FRONTEND_ORIGIN: Allowed frontend origin for CORS (default: http://localhost:5173)
- DEBUG_CHUNKS: Show retrieved chunks in console (default: true)
- ENABLE_QUERY_REWRITE: LLM query rewriting before retrieval (default: true)
- PDF_SERVICES_CLIENT_ID: Adobe PDF Services client ID
- PDF_SERVICES_CLIENT_SECRET: Adobe PDF Services client secret
"""

import os
import re
import json
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_compress import Compress
from shared import get_llm, get_embeddings, CHROMA_DIR, DATA_DIR, FIGURES_DIR

def _extract_text(content):
    """Normalize LLM response content to a plain string.
    
    Newer Gemini models can return a list of content parts instead of a
    simple string.  This handles both cases.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


DEBUG_CHUNKS = os.getenv("DEBUG_CHUNKS", "true").lower() == "true"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"

print(f"[Loaded SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}, RETRIEVAL_K: {RETRIEVAL_K}, ENABLE_QUERY_REWRITE: {ENABLE_QUERY_REWRITE}]")

ALLOWED_EXTENSIONS = {'.txt', '.pdf'}

# Initialize Flask app (API-only; frontend is separate)
app = Flask(__name__)
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
allowed_origins = [o.strip() for o in frontend_origin.split(",") if o.strip()]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
Compress(app)
print(f"[CORS allowed origins: {allowed_origins}]")


# Helpers
def format_docs_with_citations(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        chunk = meta.get("chunk", "unknown")
        page = meta.get("page")
        start_line = meta.get("start_line")
        
        # Build location string
        location_parts = []
        if page is not None:
            location_parts.append(f"Page {page}")
        if start_line is not None:
            location_parts.append(f"Line ~{start_line}")
        
        location = ", ".join(location_parts) if location_parts else f"Chunk {chunk}"

        parts.append(
            f"[{i}] Source: {source} ({location})\n{doc.page_content}"
        )

    return "\n\n".join(parts)


def _build_bounds(meta):
    """Reconstruct bounds dict from stored metadata fields."""
    x_min = meta.get("bounds_x_min")
    if x_min is None:
        return None
    return {
        "x_min": float(x_min),
        "y_min": float(meta.get("bounds_y_min", 0)),
        "x_max": float(meta.get("bounds_x_max", 0)),
        "y_max": float(meta.get("bounds_y_max", 0)),
    }


def build_sources(docs, preview_len=200):
    sources = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        page = meta.get("page")
        start_line = meta.get("start_line")
        chunk_type = meta.get("chunk_type", "text")
        figure_id = meta.get("figure_id")
        
        # Build human-readable reference with chunk type
        ref_parts = []
        if page is not None:
            ref_parts.append(f"p.{page}")
        
        # Add chunk type indicator for non-text chunks
        if chunk_type == "table":
            ref_parts.append("table")
        elif chunk_type == "figure" and figure_id:
            ref_parts.append(f"figure: {figure_id}")
        elif chunk_type == "figure":
            ref_parts.append("figure")
        elif start_line is not None:
            ref_parts.append(f"~L{start_line}")
        
        reference = ", ".join(ref_parts) if ref_parts else f"chunk {meta.get('chunk', 'unknown')}"
        
        image_url = None
        if chunk_type in ("figure", "table"):
            image_path = meta.get("image_path", "")
            if image_path and os.path.isfile(image_path):
                image_url = f"/api/figures/{os.path.basename(image_path)}"

        sources.append({
            "id": i,
            "source": meta.get("source"),
            "path": meta.get("path"),
            "reference": reference,
            "page": page,
            "line": start_line,
            "chunk": meta.get("chunk"),
            "chunk_type": chunk_type,
            "figure_id": figure_id,
            "image_url": image_url,
            "preview": doc.page_content[:preview_len],
            "bounds": _build_bounds(meta),
        })
    return sources

# Initialize the chat model
llm = get_llm()

# Load the vector database
embeddings = get_embeddings()
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

# When re-ranking is enabled, fetch more candidates for the cross-encoder to score
_FETCH_K = RETRIEVAL_K * 3 if ENABLE_RERANKING else RETRIEVAL_K

# Create vector retriever
vector_retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": _FETCH_K,
        "score_threshold": SIMILARITY_THRESHOLD
    }
)


def _build_bm25_retriever():
    """Build a BM25 keyword retriever from all documents in ChromaDB."""
    try:
        collection = vectordb._collection
        result = collection.get(include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(result["documents"], result["metadatas"])
            if text and text.strip()
        ]
        if not docs:
            return None
        bm25 = BM25Retriever.from_documents(docs, k=_FETCH_K)
        print(f"[BM25] Built keyword index from {len(docs)} documents")
        return bm25
    except Exception as e:
        print(f"[BM25] Failed to build index: {e}")
        return None


def _build_hybrid_retriever():
    """Combine vector + BM25 into an ensemble retriever."""
    bm25 = _build_bm25_retriever()
    if bm25 is None:
        print("[Hybrid] BM25 unavailable, using vector-only retrieval")
        return vector_retriever
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[0.6, 0.4]  # 60% semantic, 40% keyword
    )
    print("[Hybrid] Ensemble retriever ready (60% semantic + 40% keyword)")
    return ensemble


retriever = _build_hybrid_retriever()

# Cross-encoder re-ranker (loaded once at startup)
_cross_encoder = None
if ENABLE_RERANKING:
    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[RERANKER] Cross-encoder re-ranking enabled (ms-marco-MiniLM-L-6-v2)")
    except Exception as e:
        print(f"[RERANKER] Failed to load cross-encoder: {e}")

# Store conversation history per session (simple in-memory storage)
# For production, use session management or database
sessions = {}

SYSTEM_PROMPT = (
    "Answer ONLY using the provided context. "
    "Use chat history only to understand the question (e.g., resolve pronouns), not as a source of facts. "
    "When a Figure or Table chunk is retrieved, analyze the provided context and captions to explain its visual meaning. "
    "For tables, interpret the data structure and relationships. For figures, use the caption and surrounding context to describe what the figure shows. "
    "When an image is provided, analyze its visual structure (arrows, layers, labels, charts, diagrams) to explain the internal data flow or process described in the user's question. "
    "Cite sources like [1], [2]. "
    "If the answer is not in the context, say you don't know."
)


def get_or_create_session(session_id):
    """Get existing session or create new one with system message."""
    if session_id not in sessions:
        sessions[session_id] = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
    return sessions[session_id]


QUERY_REWRITE_PROMPT = (
    "You are a search query optimizer for a document retrieval system.\n"
    "Given a user question and recent chat history, generate 1-3 keyword-based search queries.\n\n"
    "Rules:\n"
    "- Output short KEYWORD phrases, NOT full sentences\n"
    "  Bad: 'What is the abstract of this paper?'\n"
    "  Good: 'abstract', 'transformer model architecture results'\n"
    "- Resolve pronouns using chat history (e.g. 'it' -> the specific entity)\n"
    "- Break multi-part questions into separate keyword queries\n"
    "- Use words that would actually appear in the document text\n"
    "- When asking about a section (abstract, introduction, conclusion), include\n"
    "  the section name as one query and likely content keywords as another\n"
    "- Output ONLY a JSON array of strings, nothing else\n\n"
    "Examples:\n"
    "  'what is in the abstract' -> [\"abstract\", \"propose model architecture results\"]\n"
    "  'who wrote this' -> [\"authors\", \"university department\"]\n"
    "  'how was it trained' -> [\"training procedure\", \"optimizer learning rate epochs\"]"
)


def rewrite_query(user_input, chat_history):
    """
    Use the LLM to rewrite the user's question into 1-3 optimized search queries.
    Falls back to the original question on any failure.
    """
    recent = []
    for msg in chat_history[-6:]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        recent.append(f"{role}: {msg.content[:200]}")
    history_text = "\n".join(recent) if recent else "(no prior conversation)"

    prompt = f"Chat history:\n{history_text}\n\nUser question: {user_input}"

    try:
        response = llm.invoke([
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = _extract_text(response.content).strip()

        # Strip markdown code fences if the LLM wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            queries = [q.strip() for q in queries if q.strip()]
            if queries:
                print(f"[QUERY REWRITE] {user_input!r} -> {queries}")
                return queries

        print(f"[QUERY REWRITE] Unexpected format, falling back: {raw[:200]}")
    except (json.JSONDecodeError, Exception) as e:
        print(f"[QUERY REWRITE] Failed ({e}), falling back to original query")

    return [user_input]


_CITATION_RE = re.compile(
    r'arXiv\s*(preprint)?|'
    r'\b\d{4}\.\d{4,5}\b|'
    r'proceedings\s+of|'
    r'In\s+(Advances|Proceedings)|'
    r'\bvol\.\s*\d|'
    r'\bpp\.\s*\d|'
    r'IEEE|ACM|ICML|NeurIPS|ICLR|EMNLP|ACL\b|'
    r'CoRR,\s*abs/',
    re.IGNORECASE,
)


def _is_citation_chunk(doc):
    """Detect chunks that are bibliography / reference entries."""
    text = doc.page_content
    if len(text) > 500:
        return False
    return bool(_CITATION_RE.search(text))


def retrieve_with_rewrite(user_input, chat_history):
    """
    Rewrite the user query into multiple search queries, run each through
    the hybrid retriever, deduplicate, and rank by cross-query frequency.
    Citation/reference chunks are demoted to fill only remaining slots.
    """
    if not ENABLE_QUERY_REWRITE:
        return retriever.invoke(user_input)

    queries = rewrite_query(user_input, chat_history)

    # Collect results from all queries, tracking how often each chunk appears
    seen = {}  # page_content hash -> (doc, count)
    for query in queries:
        docs = retriever.invoke(query)
        for doc in docs:
            key = hash(doc.page_content)
            if key in seen:
                seen[key] = (seen[key][0], seen[key][1] + 1)
            else:
                seen[key] = (doc, 1)

    ranked = sorted(seen.values(), key=lambda pair: pair[1], reverse=True)

    # Cross-encoder re-ranking: score each candidate against the original query
    if _cross_encoder and ranked:
        pairs = [[user_input, doc.page_content] for doc, _count in ranked]
        scores = _cross_encoder.predict(pairs)
        # Re-sort by cross-encoder score (higher = more relevant)
        ranked = [
            (doc, count)
            for (doc, count), _score in sorted(
                zip(ranked, scores), key=lambda x: x[1], reverse=True
            )
        ]
        print(f"[RERANKER] Re-ranked {len(ranked)} candidates (top score: {max(scores):.3f})")

    # Partition into priority tiers: figures/tables > content > citations
    priority_docs = []
    content_docs = []
    citation_docs = []
    for doc, count in ranked:
        chunk_type = (doc.metadata or {}).get("chunk_type")
        if chunk_type in ("figure", "table"):
            priority_docs.append((doc, count))
        elif _is_citation_chunk(doc):
            citation_docs.append((doc, count))
        else:
            content_docs.append((doc, count))

    max_results = RETRIEVAL_K * 2
    results = [doc for doc, _c in priority_docs]
    remaining = max_results - len(results)
    results.extend(doc for doc, _c in content_docs[:remaining])
    remaining = max_results - len(results)
    if remaining > 0:
        results.extend(doc for doc, _c in citation_docs[:remaining])

    pri_count = len(priority_docs)
    cit_count = len(citation_docs)
    print(f"[MULTI-QUERY] {len(queries)} queries -> {sum(c for _, c in ranked)} raw hits -> {len(results)} unique chunks (cap {max_results}, {pri_count} figures/tables prioritized, {cit_count} citations demoted)")
    return results


def process_query(user_input, session_id="default"):
    """
    Process a user query and return the response with sources.
    Supports MULTIMODAL queries: sends images to Gemini Vision API when figure chunks are retrieved.
    """
    print(f"\n[USER QUERY] {user_input}")
    
    chat_history = get_or_create_session(session_id)

    # Retrieve relevant chunks (with optional LLM query rewriting)
    relevant_docs = retrieve_with_rewrite(user_input, chat_history)

    if not relevant_docs:
        return {
            "answer": "I don't know.",
            "sources": [],
            "chunks_retrieved": 0
        }

    context = format_docs_with_citations(relevant_docs)
    sources = build_sources(relevant_docs)

    if DEBUG_CHUNKS:
        print(f"\n[Retrieved {len(relevant_docs)} relevant chunks from database]")
        print("\n[CHUNKS RETRIEVED:]")

        for i, doc in enumerate(relevant_docs, 1):
            meta = doc.metadata or {}
            src = meta.get("source", "unknown")
            path = meta.get("path", "unknown")
            chunk_id = meta.get("chunk", "unknown")
            chunk_type = meta.get("chunk_type", "text")

            print(f"\n  Chunk {i}:")
            print(f"  Source: {src}")
            print(f"  Path: {path}")
            print(f"  Chunk: {chunk_id}")
            print(f"  Type: {chunk_type}")
            if chunk_type == "figure":
                print(f"  Figure Type: {meta.get('figure_type', 'N/A')}")
                print(f"  Quality Score: {meta.get('quality_score', 'N/A')}")
                desc = meta.get('description', '')
                if desc:
                    print(f"  Description: {desc[:150]}...")
            print(f"  {doc.page_content}")

        print("[END CHUNKS]\n")

    # DETAILED DIAGNOSTIC LOGGING
    print(f"\n=== QUERY DIAGNOSTICS ===")
    print(f"Question: {user_input}")
    print(f"Retrieved {len(relevant_docs)} chunks")
    
    chunk_types = {}
    for doc in relevant_docs:
        meta = doc.metadata or {}
        ctype = meta.get('chunk_type', 'unknown')
        chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
    
    print(f"Chunk types: {chunk_types}")
    print(f"=========================\n")

    # ── LAZY VISION: detect figure chunks with images ──
    figure_images = []
    for doc in relevant_docs:
        meta = doc.metadata or {}
        if meta.get("chunk_type") == "figure":
            figure_id = meta.get('figure_id', 'Unknown')
            fig_type = meta.get('figure_type', 'unknown')
            quality = meta.get('quality_score', 0.0)
            image_path = meta.get('image_path', '')
            print(f"  [FIGURE] {figure_id} (type={fig_type}, quality={quality})")

            desc = meta.get('description', '')
            if desc:
                print(f"    Description: {desc[:100]}...")

            # Collect images that exist on disk for lazy Vision
            if image_path and os.path.exists(image_path):
                figure_images.append({
                    'figure_id': figure_id,
                    'image_path': image_path,
                })
                print(f"    Image queued for lazy Vision: {os.path.basename(image_path)}")
            else:
                print(f"    (No image file — using spatial description only)")

    # Build the query prompt — spatial descriptions are already in the context
    rag_prompt_text = f"Context from document:\n{context}\n\nQuestion: {user_input}"

    # ── DECIDE PATH: Lazy Vision (multimodal) vs Text-only ──
    if figure_images and _is_gemini_available():
        # LAZY VISION PATH: send images + question to Gemini on-the-fly
        print(f"\nLAZY VISION PATH: Sending {len(figure_images)} figure image(s) to Gemini Vision")

        # Check cache first
        cache_key = _build_vision_cache_key(figure_images, user_input)
        cached = _vision_cache.get(cache_key)
        if cached:
            print(f"  Cache hit - reusing previous Vision answer")
            response_text = cached
        else:
            # Build multimodal content: text context + images + question
            images_for_vision = _load_figure_images(figure_images)
            response_text = _query_gemini_vision(rag_prompt_text, images_for_vision, chat_history)
            # Cache ONLY successful responses (not errors)
            if not response_text.startswith("Error"):
                _vision_cache[cache_key] = response_text
                print(f"  Cached Vision answer (cache size: {len(_vision_cache)})")
            else:
                print(f"  Skipped caching error response")
    else:
        # TEXT-ONLY PATH: figure info is embedded as spatial descriptions
        if figure_images:
            print(f"\nTEXT PATH: Gemini unavailable - using spatial descriptions")
        else:
            print(f"\nTEXT PATH: No figure images - standard LLM query")
        rag_prompt = HumanMessage(content=rag_prompt_text)
        response = llm.invoke(chat_history + [rag_prompt])
        response_text = _extract_text(response.content)

    # Update chat history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(SystemMessage(content=response_text))  # Store as system message to avoid confusion

    return {
        "answer": response_text,
        "sources": sources,
        "chunks_retrieved": len(relevant_docs)
    }


# ── Lazy Vision cache (avoids duplicate API calls for same figure+question) ──
_vision_cache: dict = {}


def _build_vision_cache_key(figure_images: list, question: str) -> str:
    """Build a deterministic cache key from image paths + question."""
    import hashlib
    paths = sorted(img['image_path'] for img in figure_images)
    raw = f"{':'.join(paths)}|{question.strip().lower()}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_figure_images(figure_images: list) -> list:
    """
    Read figure image files from disk and return Base64-encoded data
    ready for Gemini Vision API.
    """
    import base64
    results = []
    for fig in figure_images:
        image_path = fig['image_path']
        figure_id = fig['figure_id']
        try:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
            
            ext = os.path.splitext(image_path)[1].lower().lstrip('.')
            mime = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
            }.get(ext, 'image/png')

            results.append({
                'data': base64.b64encode(img_bytes).decode('utf-8'),
                'mime_type': mime,
                'figure_id': figure_id,
            })
            print(f"    Loaded {figure_id}: {os.path.basename(image_path)} ({len(img_bytes)} bytes)")
        except Exception as e:
            print(f"    Failed to load {figure_id}: {e}")
    return results


def _is_gemini_available():
    """Check if Gemini API is configured."""
    google_key = os.getenv("GOOGLE_API_KEY")
    return google_key and google_key != "your_google_api_key_here"


def _query_gemini_vision(prompt_text, images, chat_history):
    """
    Query Gemini Vision API with text and images.
    
    Args:
        prompt_text: The text prompt with context and question
        images: List of dicts with 'data' (base64), 'mime_type', 'figure_id'
        chat_history: Chat history for context
    
    Returns:
        Response text from Gemini
    """
    from google import genai
    import base64
    
    # Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    # Build multimodal content parts
    content_parts = [prompt_text]
    
    for img_data in images:
        try:
            img_bytes = base64.b64decode(img_data["data"])
            mime_type = img_data.get("mime_type", "image/png")
            content_parts.append(genai.types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
            content_parts.append(f"\n[The above image is {img_data['figure_id']}]")
        except Exception as e:
            print(f"  Warning: Failed to process image {img_data['figure_id']}: {e}")
    
    # Generate response with automatic model fallback on 429
    vision_model = os.getenv("VISION_MODEL_NAME", "gemini-3.1-flash-lite-preview")
    fallback_model = "gemini-1.5-flash"
    models = [vision_model, fallback_model]
    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=content_parts,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            if model_name != models[0]:
                print(f"  Fallback to {model_name} succeeded")
            return response.text
        except Exception as e:
            error_str = str(e)
            is_429 = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            if is_429 and model_name != models[-1]:
                print(f"  {model_name} hit 429 - retrying with {models[models.index(model_name) + 1]}...")
                continue
            print(f"  Error: Gemini Vision API failed ({model_name}): {e}")
            return f"Error processing visual content: {error_str}"



# Flask Routes
@app.route('/')
def index():
    """API root endpoint."""
    return jsonify({
        "service": "RAG Chatbot API",
        "status": "ok",
        "message": "Use /api/* endpoints from the frontend app."
    })


@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint for chat queries."""
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data['message'].strip()
    session_id = data.get('session_id', 'default')

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        result = process_query(user_message, session_id)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Chat API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear_session():
    """Clear conversation history for a session."""
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default')

    if session_id in sessions:
        del sessions[session_id]

    return jsonify({"status": "cleared", "session_id": session_id})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "vector_db": os.path.exists(CHROMA_DIR)
    })


@app.route('/api/figures/<path:filename>', methods=['GET'])
def serve_figure(filename):
    """Serve extracted figure images from the assets/figures directory."""
    safe_name = os.path.basename(filename)
    if not os.path.isfile(os.path.join(FIGURES_DIR, safe_name)):
        return jsonify({"error": "Figure not found"}), 404
    return send_from_directory(FIGURES_DIR, safe_name)


@app.route('/api/documents/<path:filename>/file', methods=['GET'])
def serve_document_file(filename):
    """Serve an original document file (PDF/TXT) for the frontend PDF viewer."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DATA_DIR, safe_name)
    if not os.path.isfile(file_path):
        return jsonify({"error": "Document not found"}), 404
    return send_from_directory(DATA_DIR, safe_name)


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload a document and ingest it into the vector database."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Sanitize filename while preserving Unicode characters (e.g. Chinese)
    # secure_filename() strips non-ASCII, so we do our own sanitization
    import re
    import time
    raw_name = file.filename
    # Remove path separators and null bytes
    sanitized = re.sub(r'[/\\:\x00]', '', raw_name).strip()
    # If nothing is left (shouldn't happen), use a timestamp
    if not sanitized or sanitized.startswith('.'):
        sanitized = f"upload_{int(time.time())}{os.path.splitext(raw_name)[1]}"
    filename = sanitized
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save file to data directory
    file_path = os.path.join(DATA_DIR, filename)
    file.save(file_path)
    print(f"[UPLOAD] Saved file: {file_path}")

    try:
        # Use prepare_documents to extract text and create chunks
        # without opening a separate Chroma connection
        from ingest_single_file import prepare_documents
        documents = prepare_documents(file_path)

        # Add documents to the server's existing vector database
        # ChromaDB has a max batch size, so add in batches
        print("Storing in ChromaDB vector database...")
        BATCH_SIZE = 5000
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            vectordb.add_documents(batch)
            print(f"  -> Added batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

        # Rebuild hybrid retriever to include new documents in BM25 index
        global retriever
        retriever = _build_hybrid_retriever()

        print(f"[UPLOAD] Successfully ingested: {filename} ({len(documents)} chunks)")
        return jsonify({
            "status": "success",
            "filename": filename,
            "message": f"Document '{filename}' uploaded and ingested successfully ({len(documents)} chunks)."
        })
    except Exception as e:
        print(f"[UPLOAD ERROR] Failed to ingest {filename}: {e}")
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List documents in the data directory."""
    if not os.path.exists(DATA_DIR):
        return jsonify({"documents": []})

    documents = []
    for f in sorted(os.listdir(DATA_DIR)):
        ext = os.path.splitext(f)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            file_path = os.path.join(DATA_DIR, f)
            documents.append({
                "name": f,
                "size": os.path.getsize(file_path),
                "type": ext[1:]  # Remove the dot
            })

    return jsonify({"documents": documents})


@app.route('/api/reset-db', methods=['POST'])
def reset_database():
    """Clear the ChromaDB vector database and data directory."""
    import shutil

    errors = []

    # Clear all documents from ChromaDB via its API
    # (do NOT delete the chroma_db directory -- that causes SQLite connection errors)
    try:
        collection = vectordb._collection
        all_ids = collection.get()['ids']
        if all_ids:
            # ChromaDB has a max batch size, delete in batches
            BATCH_SIZE = 5000
            for i in range(0, len(all_ids), BATCH_SIZE):
                batch = all_ids[i:i + BATCH_SIZE]
                collection.delete(ids=batch)
            print(f"[RESET] Cleared {len(all_ids)} documents from ChromaDB")
        else:
            print("[RESET] ChromaDB already empty")
    except Exception as e:
        errors.append(f"Failed to clear ChromaDB: {str(e)}")

    # Clear uploaded data files
    if os.path.exists(DATA_DIR):
        try:
            shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR, exist_ok=True)
            print(f"[RESET] Cleared data directory: {DATA_DIR}")
        except Exception as e:
            errors.append(f"Failed to clear data directory: {str(e)}")

    if errors:
        return jsonify({"error": "; ".join(errors)}), 500

    # Rebuild hybrid retriever (BM25 index is now empty)
    global retriever
    retriever = _build_hybrid_retriever()

    return jsonify({
        "status": "success",
        "message": "Database and uploaded documents cleared successfully."
    })


def chat_cli():
    """Command-line interface for the chatbot."""
    print("RAG Chatbot initialized! Type 'quit' or 'exit' to end the conversation.")
    print("I can answer questions based on the ingested document.\n")

    session_id = "cli"

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        result = process_query(user_input, session_id)

        print(f"\nBot: {result['answer']}\n")

        if result['sources']:
            print("References:")
            for s in result['sources']:
                print(
                    f"[{s['id']}] {s['source']} ({s['reference']}) | {s['path']}"
                )
            print()


@app.route("/api/debug/search", methods=["GET"])
def debug_search():
    """Debug endpoint: raw ChromaDB similarity search with scores."""
    query = request.args.get("q", "abstract")
    k = int(request.args.get("k", "10"))
    results = vectordb.similarity_search_with_relevance_scores(query, k=k)
    output = []
    for doc, score in results:
        output.append({
            "score": round(score, 4),
            "chunk_type": doc.metadata.get("chunk_type", "?"),
            "page": doc.metadata.get("page", "?"),
            "preview": doc.page_content[:150],
        })
    print(f"\n[DEBUG SEARCH] q={query!r} k={k}")
    for i, item in enumerate(output):
        print(f"  {i+1}. score={item['score']} type={item['chunk_type']} p.{item['page']}: {item['preview'][:80]}")
    return jsonify(output)


def run_server():
    """Run the Flask web server."""
    host = os.getenv("RAG_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("RAG_SERVER_PORT", "8080"))

    print(f"\n{'='*50}")
    print(f"RAG Chatbot Server starting...")
    print(f"Open http://localhost:{port} in your browser")
    print(f"{'='*50}\n")

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    import sys

    if not os.path.exists(CHROMA_DIR):
        print("Error: Vector database not found!")
        print("Please run 'python ingest.py' first to ingest your document.")
        sys.exit(1)

    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        chat_cli()
    else:
        run_server()
