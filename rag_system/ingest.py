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

def split_text(text, chunk_size=1000, chunk_overlap=300):
    """
    Split text into overlapping chunks.
    
    Increased overlap to 300 to ensure figure references (like 'see Figure 2')
    are captured alongside descriptive text.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk (default: 1000)
        chunk_overlap: Overlap between chunks (default: 300)
    
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
                    # Extract structured chunks from PDF
                    structured_chunks = extract_text_from_pdf(file_path)
                    
                    # Process each structured chunk
                    for chunk_data in structured_chunks:
                        chunk_type = chunk_data.get('type', 'text')
                        page_num = chunk_data.get('page', 1)
                        content = chunk_data.get('content', '')
                        
                        # CRITICAL SAFETY: Implement fallback for empty content
                        if not content or not content.strip():
                            # Fallback for figures
                            if chunk_type == 'figure':
                                figure_id = chunk_data.get('figure_id', f'Figure_{page_num}')
                                content = f'Visual element {figure_id} on page {page_num}'
                                print(f"  Warning: Using fallback content for {figure_id}")
                            else:
                                print(f"  Warning: Skipping empty {chunk_type} chunk on page {page_num}")
                                continue
                        
                        # Handle different chunk types
                        if chunk_type == 'text':
                            # Split text chunks further for better retrieval
                            text_splits = split_text(content)
                            
                            for i, split_dict in enumerate(text_splits):
                                split_content = split_dict.get('text', '').strip()
                                if not split_content:  # Skip empty splits
                                    continue
                                
                                # Calculate approximate line number from character position
                                lines_before = content[:split_dict['start_pos']].count('\n')
                                start_line = lines_before + 1
                                
                                documents.append(
                                    Document(
                                        page_content=split_content,
                                        metadata={
                                            "source": file,
                                            "path": file_path,
                                            "chunk": i,
                                            "page": page_num,
                                            "chunk_type": "text",
                                            "start_line": start_line if page_num else None
                                        }
                                    )
                                )
                        
                        elif chunk_type == 'table':
                            # Store tables as single documents (no splitting)
                            documents.append(
                                Document(
                                    page_content=content.strip(),
                                    metadata={
                                        "source": file,
                                        "path": file_path,
                                        "chunk": 0,
                                        "page": page_num,
                                        "chunk_type": "table"
                                    }
                                )
                            )
                        
                        elif chunk_type == 'figure':
                            # Store Pydantic-validated figures with rich descriptions
                            figure_id = chunk_data.get('figure_id', f'Figure_{page_num}')
                            figure_type = chunk_data.get('figure_type', 'unknown')
                            quality_score = chunk_data.get('quality_score', 0.0)
                            description = chunk_data.get('description', '')
                            
                            documents.append(
                                Document(
                                    page_content=content.strip(),
                                    metadata={
                                        "source": file,
                                        "path": file_path,
                                        "chunk": 0,
                                        "page": page_num,
                                        "chunk_type": "figure",
                                        "figure_id": figure_id,
                                        "figure_type": figure_type,
                                        "quality_score": quality_score,
                                        "description": description,
                                        "image_path": chunk_data.get('image_path', ''),
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
                                "chunk_type": "text",
                                "start_line": start_line  # Line number in text file
                            }
                        )
                    )

    print(f"Total chunks created: {len(documents)}")
    
    # Count chunk types
    text_chunks = sum(1 for d in documents if d.metadata.get('chunk_type') == 'text')
    table_chunks = sum(1 for d in documents if d.metadata.get('chunk_type') == 'table')
    figure_chunks = sum(1 for d in documents if d.metadata.get('chunk_type') == 'figure')
    
    print(f"  - Text chunks: {text_chunks}")
    print(f"  - Table chunks: {table_chunks}")
    print(f"  - Figure chunks: {figure_chunks}")

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
