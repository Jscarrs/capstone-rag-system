import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"

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

def split_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks.

    Args:
        text: The text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks

def ingest_all_documents(data_dir=DATA_DIR):
    documents = []

    print(f"Scanning folder: {data_dir}")

    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith((".txt")):
                continue

            file_path = os.path.join(root, file)
            print(f"Loading file: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = split_text(text)

            for i, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file,
                            "path": file_path,
                            "chunk": i
                        }
                    )
                )

    print(f"Total chunks created: {len(documents)}")

    embeddings = get_embeddings()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("Ingestion complete")
    return vectordb

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
    else:
        ingest_all_documents(DATA_DIR)