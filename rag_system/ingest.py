import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
USE_MARKER_OCR = os.getenv("USE_MARKER_OCR", os.getenv("USE_ADVANCED_OCR", "false")).lower() == "true"
USE_ADOBE_OCR = os.getenv("USE_ADOBE_OCR", "false").lower() == "true"

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

def ingest_all_documents(data_dir=DATA_DIR):
    from pdf_processor import extract_text_from_pdf, is_pdf_file
    
    documents = []

    print(f"Scanning folder: {data_dir}")

    for root, _, files in os.walk(data_dir):
        for file in files:
            # Support both .txt and .pdf files
            if not file.lower().endswith((".txt", ".pdf")):
                continue

            file_path = os.path.join(root, file)
            print(f"Loading file: {file_path}")

            # Handle PDF files
            if is_pdf_file(file_path):
                try:
                    page_data = extract_text_from_pdf(file_path, force_marker=USE_MARKER_OCR, use_adobe=USE_ADOBE_OCR)
                    
                    # Process each page
                    for page_dict in page_data:
                        page_num = page_dict['page']
                        page_text = page_dict['text']
                        
                        # Split page text into chunks
                        chunks = split_text(page_text)
                        
                        for i, chunk_dict in enumerate(chunks):
                            # Calculate approximate line number from character position
                            lines_before = page_text[:chunk_dict['start_pos']].count('\n')
                            start_line = lines_before + 1
                            
                            documents.append(
                                Document(
                                    page_content=chunk_dict['text'],
                                    metadata={
                                        "source": file,
                                        "path": file_path,
                                        "chunk": i,
                                        "page": page_num,  # Can be None for OCR
                                        "start_line": start_line if page_num else None
                                    }
                                )
                            )
                            
                except Exception as e:
                    print(f"  Error processing PDF {file_path}: {str(e)}")
                    continue
                    
            # Handle text files
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = split_text(text)

                for i, chunk_dict in enumerate(chunks):
                    # Calculate line number from character position
                    lines_before = text[:chunk_dict['start_pos']].count('\n')
                    start_line = lines_before + 1
                    
                    documents.append(
                        Document(
                            page_content=chunk_dict['text'],
                            metadata={
                                "source": file,
                                "path": file_path,
                                "chunk": i,
                                "start_line": start_line  # Line number in text file
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
