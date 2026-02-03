"""
PDF Processing Module with Hybrid Extraction

This module provides intelligent PDF text extraction with a hybrid approach:
1. Fast standard extraction using pdfplumber (for simple text-based PDFs)
2. Advanced Marker-based extraction (for complex/scanned PDFs)

The hybrid approach optimizes performance by using fast extraction when
possible and only falling back to Marker when needed.
"""

import os
from typing import Tuple, Optional
import re


def detect_pdf_complexity(pdf_path: str, max_pages_to_check: int = 5) -> dict:
    """
    Detect if a PDF contains complex elements like tables, images, or math.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages_to_check: Number of pages to sample (default: 5)
        
    Returns:
        Dict with complexity indicators: {
            'has_tables': bool,
            'has_images': bool,
            'has_math': bool,
            'is_complex': bool,
            'reason': str
        }
    """
    try:
        import pdfplumber
        
        has_tables = False
        has_images = False
        has_math = False
        reasons = []
        
        # Common math symbols and indicators
        math_symbols = ['∫', '∑', '∏', '√', '∂', '∆', 'α', 'β', 'γ', 'δ', 'θ', 'λ', 'μ', 'π', 'σ', 
                       '≤', '≥', '±', '∞', '≈', '≠', '∈', '∉', '⊂', '⊃', '∀', '∃']
        
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = min(max_pages_to_check, len(pdf.pages))
            
            for i in range(pages_to_check):
                page = pdf.pages[i]
                
                # Check for tables
                if not has_tables:
                    tables = page.find_tables()
                    if len(tables) > 0:
                        has_tables = True
                        reasons.append(f"tables on page {i+1}")
                
                # Check for images/diagrams
                if not has_images:
                    if len(page.images) > 0:
                        has_images = True
                        reasons.append(f"images/diagrams on page {i+1}")
                
                # Check for math symbols
                if not has_math:
                    page_text = page.extract_text() or ""
                    if any(symbol in page_text for symbol in math_symbols):
                        has_math = True
                        reasons.append(f"math symbols on page {i+1}")
                    
                    # Also check for common LaTeX patterns (some PDFs preserve these)
                    latex_patterns = [r'\$[^$]+\$', r'\\frac', r'\\sum', r'\\int', r'\\sqrt']
                    if any(re.search(pattern, page_text) for pattern in latex_patterns):
                        has_math = True
                        reasons.append(f"LaTeX notation on page {i+1}")
                
                # Early exit if we found complexity
                if has_tables or has_images or has_math:
                    break
        
        is_complex = has_tables or has_images or has_math
        reason = "; ".join(reasons) if reasons else "no complex elements detected"
        
        return {
            'has_tables': has_tables,
            'has_images': has_images,
            'has_math': has_math,
            'is_complex': is_complex,
            'reason': reason
        }
        
    except Exception as e:
        print(f"  ⚠ Complexity detection failed: {str(e)}")
        # If detection fails, assume simple
        return {
            'has_tables': False,
            'has_images': False,
            'has_math': False,
            'is_complex': False,
            'reason': 'detection failed - assuming simple'
        }


def try_standard_extraction(pdf_path: str) -> Tuple[Optional[list], bool]:
    """
    Attempt fast text extraction using pdfplumber with page number tracking.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (page_data, success)
        - page_data: List of dicts with 'page' and 'text' keys, or None if failed
        - success: True if extraction was successful and yielded enough content
    """
    try:
        import pdfplumber
        
        page_data = []
        total_chars = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    page_data.append({
                        'page': page_num,
                        'text': page_text
                    })
                    total_chars += len(page_text)
        
        # Check if we got sufficient content
        # Heuristic: at least 100 characters suggests real text content
        if total_chars > 100:
            print(f"  ✓ Standard extraction successful ({total_chars} chars from {len(page_data)} pages)")
            return page_data, True
        else:
            print(f"  ✗ Standard extraction yielded insufficient text ({total_chars} chars)")
            return None, False
            
    except Exception as e:
        print(f"  ✗ Standard extraction failed: {str(e)}")
        return None, False




def extract_with_marker(pdf_path: str) -> Optional[str]:

    """
    Extract text from PDF using Marker (advanced OCR with layout preservation).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted markdown text or None if failed
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        
        print(f"  → Using Marker for complex PDF processing...")
        
        # Initialize Marker converter
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        
        # Convert PDF
        rendered = converter(pdf_path)
        
        # Extract text from rendered output
        text, _, images = text_from_rendered(rendered)
        
        if text and len(text.strip()) > 0:
            print(f"  ✓ Marker extraction successful ({len(text)} chars)")
            return text
        else:
            print(f"  ✗ Marker extraction failed to produce text")
            return None
            
    except ImportError as e:
        print(f"  ✗ Marker not installed. Run: pip install marker-pdf")
        print(f"     Error: {str(e)}")
        return None
    except Exception as e:
        print(f"  ✗ Marker extraction failed: {str(e)}")
        return None


def extract_text_from_pdf(pdf_path: str, force_marker: bool = False, auto_detect: bool = True):
    """
    Extract text from PDF using smart hybrid approach with complexity detection.
    
    This function intelligently chooses the best extraction method:
    1. Detects PDF complexity (tables, images, math) if auto_detect=True
    2. Uses Marker for complex PDFs, standard extraction for simple PDFs
    3. Falls back to Marker if standard extraction fails
    
    Args:
        pdf_path: Path to the PDF file
        force_marker: If True, skip detection and use Marker directly
        auto_detect: If True, detect complexity and choose method automatically
        
    Returns:
        List of dicts with 'page' and 'text' keys for page-by-page content,
        or a single dict with 'page': None and 'text': full_text for Marker extraction
        
    Raises:
        ValueError: If both extraction methods fail
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"\nProcessing PDF: {os.path.basename(pdf_path)}")
    
    # Smart detection: check if PDF is complex
    use_marker = force_marker
    
    if not force_marker and auto_detect:
        print("  🔍 Detecting PDF complexity...")
        complexity = detect_pdf_complexity(pdf_path)
        
        if complexity['is_complex']:
            use_marker = True
            print(f"  ✓ Complex PDF detected ({complexity['reason']})")
            print(f"  → Using Marker for better quality")
        else:
            print(f"  ✓ Simple PDF detected ({complexity['reason']})")
            print(f"  → Using fast standard extraction")
    
    # Use Marker if determined to be complex or forced
    if use_marker:
        text = extract_with_marker(pdf_path)
        if text:
            return [{'page': None, 'text': text}]
        else:
            # Marker failed, try fallback to standard
            print("  ⚠ Marker failed, falling back to standard extraction...")
            page_data, success = try_standard_extraction(pdf_path)
            if success and page_data:
                return page_data
            else:
                raise ValueError(f"Both Marker and standard extraction failed for {pdf_path}")
    
    # Try standard extraction for simple PDFs
    page_data, success = try_standard_extraction(pdf_path)
    if success and page_data:
        return page_data
    
    # Standard extraction failed, try Marker as fallback
    print("  ⚠ Standard extraction failed, trying Marker as fallback...")
    text = extract_with_marker(pdf_path)
    
    if text:
        return [{'page': None, 'text': text}]
    else:
        raise ValueError(
            f"Failed to extract text from PDF: {pdf_path}\n"
            f"Both standard and Marker extraction failed. "
            f"Ensure dependencies are installed: pip install pdfplumber marker-pdf"
        )


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
        page_data = extract_text_from_pdf(pdf_file)
        print("\n" + "="*80)
        print("EXTRACTED TEXT:")
        print("="*80)
        
        for page_dict in page_data[:3]:  # Show first 3 pages
            page_num = page_dict['page']
            text = page_dict['text']
            
            if page_num:
                print(f"\n--- Page {page_num} ---")
            else:
                print(f"\n--- Full Document (OCR) ---")
            
            print(text[:500])  # Show first 500 chars per page
            if len(text) > 500:
                print(f"... ({len(text)} chars on this page)")
        
        if len(page_data) > 3:
            print(f"\n... ({len(page_data) - 3} more pages not shown)")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
