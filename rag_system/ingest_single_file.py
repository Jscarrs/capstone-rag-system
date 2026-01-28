import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

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

def ingest_document(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Load a text file, split it into chunks, embed, and store in ChromaDB.

    Args:
        file_path: Path to the text file
        chunk_size: Size of each text chunk (in characters)
        chunk_overlap: Overlap between chunks to maintain context
    """
    print(f"Loading document from {file_path}...")

    # Load the document
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded document")
    print(f"Total characters: {len(text)}")

    # Split the text into chunks
    print(f"\nSplitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_chunks = split_text(text, chunk_size, chunk_overlap)

    # Convert to Document objects
    documents = [Document(page_content=chunk, metadata={"source": file_path}) for chunk in text_chunks]

    print(f"Created {len(documents)} chunks")

    # Create embeddings
    print("\nCreating embeddings...")
    embeddings = get_embeddings()

    # Create and persist the vector database
    print("Storing in ChromaDB vector database...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print(f"\n✓ Successfully ingested document!")
    print(f"✓ Vector database saved to ./rag_system/chroma_db")
    print(f"✓ Total chunks stored: {len(documents)}")

    return vectordb

if __name__ == "__main__":
    # Example usage
    file_path = "./data/book.txt"

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please place your text file at ./rag_system/data/book.txt")
    else:
        ingest_document(file_path)
