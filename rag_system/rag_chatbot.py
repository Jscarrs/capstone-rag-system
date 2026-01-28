import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

DEBUG_CHUNKS = True

# Load environment variables
load_dotenv()

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

        parts.append(
            f"[{i}] Source: {source}, Chunk: {chunk}\n{doc.page_content}"
        )

    return "\n\n".join(parts)


def build_sources(docs, preview_len=200):
    sources = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        sources.append({
            "id": i,
            "source": meta.get("source"),
            "path": meta.get("path"),
            "chunk": meta.get("chunk"),
            "preview": doc.page_content[:preview_len]
        })
    return sources

# Initialize the chat model
llm = get_llm()

# Load the vector database
embeddings = get_embeddings()
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

# Create a retriever from the vector database
retriever = vectordb.as_retriever(
search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3
    }
)

# Store conversation history
chat_history = []

def chat():
    print("RAG Chatbot initialized! Type 'quit' or 'exit' to end the conversation.")
    print("I can answer questions based on the ingested document.\n")

    # Add system message to set context
    system_message = SystemMessage(
        content=("Answer ONLY using the provided context. "
            "Cite sources like [1], [2]. "
            "If the answer is not in the context, say you don't know."
        )
    )
    chat_history.append(system_message)

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Retrieve relevant chunks from vector database
        relevant_docs = retriever.invoke(user_input)

        if not relevant_docs:
            print("\nBot: I don't know.\n")
            continue

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

        # Create a message with context
        contextualized_message = HumanMessage(
            content=f"Context from document:\n{context}\n\nQuestion: {user_input}"
        )

        # Add to history
        chat_history.append(contextualized_message)

        # Get response from LLM
        response = llm.invoke(chat_history)

        # Add AI response to history
        chat_history.append(response)

        print(f"\nBot: {response.content}\n")
        print("References:")
        for s in sources:
            print(
                f"[{s['id']}] {s['source']} "
                f"(chunk {s['chunk']}) | {s['path']}"
            )
        print()

if __name__ == "__main__":
    if not os.path.exists("./chroma_db"):
        print("Error: Vector database not found!")
        print("Please run 'python ingest.py' first to ingest your document.")
    else:
        chat()
