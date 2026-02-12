"""
Adobe PDF Services OCR Module

Requirements:
- pdfservices-sdk Python package
- Environment variables: PDF_SERVICES_CLIENT_ID, PDF_SERVICES_CLIENT_SECRET

Provides text extraction from PDFs using Adobe PDF Services Extract API.
Returns page-level text in the same format as pdf_processor.py:
  [{'page': int|None, 'text': str}, ...]
"""

import os
import json
from typing import Optional


def is_adobe_ocr_available() -> bool:
    """
    Check if Adobe PDF Services credentials are configured.

    Returns:
        True if both PDF_SERVICES_CLIENT_ID and PDF_SERVICES_CLIENT_SECRET are set
    """
    client_id = os.getenv("PDF_SERVICES_CLIENT_ID", "")
    client_secret = os.getenv("PDF_SERVICES_CLIENT_SECRET", "")
    return bool(client_id) and bool(client_secret)


def extract_text_with_adobe(pdf_path: str) -> list:
    """
    Extract text from a PDF using Adobe PDF Services Extract API.

    Uploads the PDF to Adobe, runs ExtractPDFJob with TEXT extraction,
    and parses the structured JSON result into page-level text.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with 'page' and 'text' keys, matching pdf_processor.py format

    Raises:
        ValueError: If credentials are not configured or extraction fails
        FileNotFoundError: If the PDF file does not exist
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not is_adobe_ocr_available():
        raise ValueError(
            "Adobe PDF Services credentials not configured. "
            "Set PDF_SERVICES_CLIENT_ID and PDF_SERVICES_CLIENT_SECRET in .env"
        )

    # Import Adobe SDK modules
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

    print(f"  -> Adobe PDF Services: extracting text from {os.path.basename(pdf_path)}...")

    # Read the PDF file
    with open(pdf_path, 'rb') as f:
        input_stream = f.read()

    # Setup credentials
    credentials = ServicePrincipalCredentials(
        client_id=os.getenv('PDF_SERVICES_CLIENT_ID'),
        client_secret=os.getenv('PDF_SERVICES_CLIENT_SECRET')
    )

    # Create PDF Services instance and upload
    pdf_services = PDFServices(credentials=credentials)
    input_asset = pdf_services.upload(
        input_stream=input_stream,
        mime_type=PDFServicesMediaType.PDF.mime_type
    )

    # Configure extraction for TEXT only
    extract_params = ExtractPDFParams(
        elements_to_extract=[ExtractElementType.TEXT]
    )

    # Create and submit the extraction job
    extract_job = ExtractPDFJob(
        input_asset=input_asset,
        extract_pdf_params=extract_params
    )
    location = pdf_services.submit(extract_job)
    pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

    # Parse the structured JSON result into page-level text
    # get_content_json() may return bytes or dict depending on SDK version
    content_json = pdf_services_response.get_result().get_content_json()
    if isinstance(content_json, bytes):
        content_json = json.loads(content_json)
    page_data = _parse_extract_json(content_json)

    total_chars = sum(len(p['text']) for p in page_data)
    print(f"  -> Adobe extraction complete ({total_chars} chars from {len(page_data)} pages)")

    return page_data


def _parse_extract_json(content_json: dict) -> list:
    """
    Parse Adobe Extract API JSON response into page-level text.

    The Extract API returns a JSON structure with elements that have
    page references. This function groups text elements by page number.

    Args:
        content_json: The JSON dict from ExtractPDFResult.get_content_json()

    Returns:
        List of dicts with 'page' and 'text' keys
    """
    elements = content_json.get("elements", [])

    # Group text by page number
    pages = {}
    for element in elements:
        # Each element has a "Path" like "//Document/P", "//Document/P[2]", etc.
        # and optionally a "Page" index (0-based)
        page_index = element.get("Page", 0)
        page_num = page_index + 1  # Convert to 1-based

        text = element.get("Text", "")
        if not text:
            continue

        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(text)

    # Build page_data list sorted by page number
    page_data = []
    for page_num in sorted(pages.keys()):
        combined_text = "\n".join(pages[page_num])
        page_data.append({
            'page': page_num,
            'text': combined_text
        })

    # If no pages were found, return the raw content as a single entry
    if not page_data:
        print("  -> Warning: No text elements found in Adobe Extract response")
        return [{'page': None, 'text': ''}]

    return page_data
