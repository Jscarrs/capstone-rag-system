import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
USE_ADVANCED_OCR = os.getenv("USE_ADVANCED_OCR", "false").lower() == "true"

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
    Split text into overlapping chunks with position tracking.

    Args:
        text: The text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of dicts with 'text' and 'start_pos' keys
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            'text': chunk,
            'start_pos': start  # Character position in original text
        })
        start += chunk_size - chunk_overlap

    return chunks

def ingest_document(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Load a document (text or PDF), split it into chunks, embed, and store in ChromaDB.

    Args:
        file_path: Path to the document file (.txt or .pdf)
        chunk_size: Size of each text chunk (in characters)
        chunk_overlap: Overlap between chunks to maintain context
    """
    from pdf_processor import extract_text_from_pdf, is_pdf_file
    
    print(f"Loading document from {file_path}...")

    documents = []
    file_name = os.path.basename(file_path)
    print(f"\nSplitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")

    # Handle PDF files
    if is_pdf_file(file_path):
        page_data = extract_text_from_pdf(file_path, force_marker=USE_ADVANCED_OCR)

        for page_dict in page_data:
            page_num = page_dict['page']
            page_text = page_dict['text']

            chunks = split_text(page_text, chunk_size, chunk_overlap)

            for i, chunk_dict in enumerate(chunks):
                lines_before = page_text[:chunk_dict['start_pos']].count('\n')
                start_line = lines_before + 1

                documents.append(
                    Document(
                        page_content=chunk_dict['text'],
                        metadata={
                            "source": file_name,
                            "path": file_path,
                            "chunk": i,
                            "page": page_num,  # Can be None for OCR
                            "start_line": start_line if page_num else None
                        }
                    )
                )

        total_chars = sum(len(page_dict.get("text", "")) for page_dict in page_data)

    # Handle text files
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = split_text(text, chunk_size, chunk_overlap)

        for i, chunk_dict in enumerate(chunks):
            lines_before = text[:chunk_dict['start_pos']].count('\n')
            start_line = lines_before + 1

            documents.append(
                Document(
                    page_content=chunk_dict['text'],
                    metadata={
                        "source": file_name,
                        "path": file_path,
                        "chunk": i,
                        "start_line": start_line
                    }
                )
            )

        total_chars = len(text)

    print("Loaded document")
    print(f"Total characters: {total_chars}")
    print(f"Created {len(documents)} chunks")

    # Create embeddings
    print("\nCreating embeddings...")
    embeddings = get_embeddings()

    # Create and persist the vector database
    print("Storing in ChromaDB vector database...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"\n✓ Successfully ingested document!")
    print(f"✓ Vector database saved to {CHROMA_DIR}")
    print(f"✓ Total chunks stored: {len(documents)}")

    return vectordb

if __name__ == "__main__":
    # Example usage - supports both .txt and .pdf files
    file_path = os.path.join(SCRIPT_DIR, "data", "book.txt")  # Or point to any .txt/.pdf file

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print(f"Please place your document at {os.path.join(SCRIPT_DIR, 'data')}")
        print("Supported formats: .txt, .pdf")
    else:
        ingest_document(file_path)
