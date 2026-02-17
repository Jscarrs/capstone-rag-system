"""
PDF Processing Module - Adobe-Only Extraction

This module provides PDF text extraction using Adobe PDF Services Extract API exclusively.
All PDFs are processed through Adobe's cloud-based extraction service, which provides
structured content including text, tables, and figures.

Environment Variables:
- PDF_SERVICES_CLIENT_ID: Adobe API client ID (required)
- PDF_SERVICES_CLIENT_SECRET: Adobe API client secret (required)
"""

import os
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract structured content from PDF using Adobe PDF Services Extract API.
    
    This function delegates to adobe_ocr.py for all PDF processing.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of structured chunks with type information:
        [
            {'type': 'text', 'page': int, 'content': str, 'order': int},
            {'type': 'table', 'page': int, 'content': str, 'order': int},
            {'type': 'figure', 'page': int, 'content': str, 'order': int, 'figure_id': str},
            ...
        ]
        
    Raises:
        ValueError: If Adobe credentials are not configured or extraction fails
        FileNotFoundError: If the PDF file does not exist
    """
    from adobe_ocr import extract_text_with_adobe, is_adobe_ocr_available
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not is_adobe_ocr_available():
        raise ValueError(
            "Adobe PDF Services credentials not configured. "
            "Set PDF_SERVICES_CLIENT_ID and PDF_SERVICES_CLIENT_SECRET in .env file.\n"
            "This system requires Adobe PDF Services for all PDF processing."
        )
    
    print(f"\nProcessing PDF: {os.path.basename(pdf_path)}")
    return extract_text_with_adobe(pdf_path)


def is_pdf_file(file_path: str) -> bool:
    """
    Check if a file is a PDF based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file has .pdf extension
    """
    return file_path.lower().endswith('.pdf')


if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <pdf_file>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    try:
        chunks = extract_text_from_pdf(pdf_file)
        print("\n" + "="*80)
        print("EXTRACTED STRUCTURED CONTENT:")
        print("="*80)
        
        for chunk in chunks[:10]:  # Show first 10 chunks
            chunk_type = chunk['type']
            page_num = chunk['page']
            content = chunk['content']
            
            print(f"\n--- {chunk_type.upper()} (Page {page_num}) ---")
            
            # Show preview of content
            preview = content[:300] if len(content) > 300 else content
            print(preview)
            if len(content) > 300:
                print(f"... ({len(content)} chars total)")
        
        if len(chunks) > 10:
            print(f"\n... ({len(chunks) - 10} more chunks not shown)")
        
        # Summary
        text_count = sum(1 for c in chunks if c['type'] == 'text')
        table_count = sum(1 for c in chunks if c['type'] == 'table')
        figure_count = sum(1 for c in chunks if c['type'] == 'figure')
        
        print("\n" + "="*80)
        print(f"SUMMARY: {text_count} text chunks, {table_count} tables, {figure_count} figures")
        print("="*80)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
