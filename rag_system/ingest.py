import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from shared import get_embeddings, split_text, DATA_DIR, CHROMA_DIR

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
