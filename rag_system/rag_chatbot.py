"""
RAG Chatbot Server

Requirements:
- Flask for web server
- LangChain for RAG pipeline
- ChromaDB for vector storage
- Environment variables for LLM configuration

Environment Variables:
- LMSTUDIO_BASE_URL: Local LM Studio server URL
- OPENAI_API_KEY: OpenAI API key
- GOOGLE_API_KEY: Google Gemini API key
- USE_LOCAL_EMBEDDINGS: Use local HuggingFace embeddings
- RAG_SERVER_HOST: Server host (default: 0.0.0.0)
- RAG_SERVER_PORT: Server port (default: 8080)
- DEBUG_CHUNKS: Show retrieved chunks in console (default: true)
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Load environment variables from parent directory
# (since .env is in project root, not in rag_system folder)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)

DEBUG_CHUNKS = os.getenv("DEBUG_CHUNKS", "true").lower() == "true"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))  # Configurable threshold

print(f"[Loaded SIMILARITY_THRESHOLD: {SIMILARITY_THRESHOLD}]")  # Debug output

# Get directory of this script for static file serving
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")

# Initialize Flask app
app = Flask(__name__, static_folder=os.path.join(SCRIPT_DIR, 'static'))
CORS(app)


def get_llm():
    """
    Initialize the LLM based on available configuration.
    Priority: LM Studio (local) > OpenAI > Google Gemini
    """
    lmstudio_url = os.getenv("LMSTUDIO_BASE_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    # LM Studio (local, no API key needed)
    if lmstudio_url:
        from langchain_openai import ChatOpenAI
        print(f"[Using LM Studio at {lmstudio_url}]")
        return ChatOpenAI(
            base_url=lmstudio_url,
            api_key="lm-studio",  # LM Studio doesn't need a real key
            temperature=0.7
        )
    elif openai_key and openai_key != "your_openai_api_key_here":
        from langchain_openai import ChatOpenAI
        print("[Using OpenAI GPT-3.5-turbo]")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=openai_key
        )
    elif google_key and google_key != "your_google_api_key_here":
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("[Using Google Gemini 2.5 Flash]")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=google_key
        )
    else:
        raise ValueError(
            "No LLM configured. Set LMSTUDIO_BASE_URL, OPENAI_API_KEY, or GOOGLE_API_KEY in your .env file."
        )

def get_embeddings():
    """
    Initialize embeddings based on available configuration.
    Priority: HuggingFace (local) > OpenAI > Google Gemini
    """
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    lmstudio_url = os.getenv("LMSTUDIO_BASE_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    # Local embeddings with HuggingFace (free, no API key)
    if use_local or lmstudio_url:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("[Using Local HuggingFace Embeddings (all-MiniLM-L6-v2)]")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    elif openai_key and openai_key != "your_openai_api_key_here":
        from langchain_openai import OpenAIEmbeddings
        print("[Using OpenAI Embeddings]")
        return OpenAIEmbeddings(openai_api_key=openai_key)
    elif google_key and google_key != "your_google_api_key_here":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        print("[Using Google Gemini Embeddings]")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_key
        )
    else:
        raise ValueError(
            "No embeddings configured. Set USE_LOCAL_EMBEDDINGS=true, OPENAI_API_KEY, or GOOGLE_API_KEY in your .env file."
        )

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


def build_sources(docs, preview_len=200):
    sources = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        page = meta.get("page")
        start_line = meta.get("start_line")
        
        # Build human-readable reference
        ref_parts = []
        if page is not None:
            ref_parts.append(f"p.{page}")
        if start_line is not None:
            ref_parts.append(f"~L{start_line}")
        
        reference = ", ".join(ref_parts) if ref_parts else f"chunk {meta.get('chunk', 'unknown')}"
        
        sources.append({
            "id": i,
            "source": meta.get("source"),
            "path": meta.get("path"),
            "reference": reference,  # New: human-readable reference
            "page": page,
            "line": start_line,
            "chunk": meta.get("chunk"),
            "preview": doc.page_content[:preview_len]
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

# Create a retriever from the vector database
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": SIMILARITY_THRESHOLD
    }
)

# Store conversation history per session (simple in-memory storage)
# For production, use session management or database
sessions = {}

SYSTEM_PROMPT = (
    "Answer ONLY using the provided context. "
    "Use chat history only to understand the question (e.g., resolve pronouns), not as a source of facts. "
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


def process_query(user_input, session_id="default"):
    """
    Process a user query and return the response with sources.
    Used by both CLI and web server.
    """
    print(f"\n[USER QUERY] {user_input}")
    
    chat_history = get_or_create_session(session_id)

    # Retrieve relevant chunks from vector database
    relevant_docs = retriever.invoke(user_input)

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

            print(f"\n  Chunk {i}:")
            print(f"  Source: {src}")
            print(f"  Path: {path}")
            print(f"  Chunk: {chunk_id}")
            print(f"  {doc.page_content}")

        print("[END CHUNKS]\n")

    # Inject retrieved context for this turn without persisting it in session history.
    rag_prompt = HumanMessage(
        content=f"Context from document:\n{context}\n\nQuestion: {user_input}"
    )

    response = llm.invoke(chat_history + [rag_prompt])

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(response)

    return {
        "answer": response.content,
        "sources": sources,
        "chunks_retrieved": len(relevant_docs)
    }


# Flask Routes
@app.route('/')
def index():
    """Serve the frontend HTML page."""
    return send_from_directory(app.static_folder, 'index.html')


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
