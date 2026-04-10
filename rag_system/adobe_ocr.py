"""
Adobe PDF Services OCR Module — Strict Adobe-Only, Pydantic-Validated Edition

This module uses the Adobe PDF Extract API as the SOLE engine for PDF OCR and
structural analysis. No other PDF libraries (PyMuPDF, pdfplumber, pypdf) are used.

Capabilities:
  1. Semantic & Structural text extraction with page/bounds metadata
  2. Table extraction from Adobe element text (cell content from TD/TH elements).
     Rendition images are saved for on-demand Gemini Vision analysis at query time.
  3. Figure Renditions → physical .png files saved to assets/figures/
  4. Pydantic-Validated Figures → quality scoring, false positive filtering,
     and spatial text descriptions stored as searchable text
  5. Zero-Null Policy → page_content is NEVER empty
  6. Multimodal Integration Helper → Base64 encoder for LLM prompts
  7. Heading Merging → section headings (detected via Adobe's structural Path)
     are merged into the following body text chunk on the same page
  [
    {'type': 'text',   'page': int, 'content': str, 'order': int,
     'bounds': [x_min, y_min, x_max, y_max]},
    {'type': 'table',  'page': int, 'content': str, 'order': int,
     'bounds': [x_min, y_min, x_max, y_max], 'image_path': str},
    {'type': 'figure', 'page': int, 'content': str (hybrid),   'order': int,
     'figure_id': str, 'image_path': str|None},
  ]
"""

import os
import re
import json
import base64
import zipfile
import shutil
import tempfile
from typing import Optional, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
FIGURES_DIR = os.path.join(ASSETS_DIR, "figures")


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def is_adobe_ocr_available() -> bool:
    """Check if Adobe PDF Services credentials are configured."""
    client_id = os.getenv("PDF_SERVICES_CLIENT_ID", "")
    client_secret = os.getenv("PDF_SERVICES_CLIENT_SECRET", "")
    return bool(client_id) and bool(client_secret)


def extract_text_with_adobe(pdf_path: str) -> list:
    """
    Extract structured content from a PDF using Adobe PDF Services Extract API.

    Extracts TEXT, TABLES, and FIGURE RENDITIONS. Saves figure images to
    assets/figures/ and returns structured chunks with hybrid figure content.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of structured chunk dicts (text / table / figure).

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        ValueError: If Adobe credentials are missing.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not is_adobe_ocr_available():
        raise ValueError(
            "Adobe PDF Services credentials not configured. "
            "Set PDF_SERVICES_CLIENT_ID and PDF_SERVICES_CLIENT_SECRET in .env"
        )

    # ── Adobe SDK imports ──
    from adobe.pdfservices.operation.auth.service_principal_credentials import (
        ServicePrincipalCredentials,
    )
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import (
        ExtractElementType,
    )
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import (
        ExtractRenditionsElementType,
    )
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import (
        ExtractPDFParams,
    )
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import (
        ExtractPDFResult,
    )

    pdf_basename = os.path.basename(pdf_path)
    print(f"  -> Adobe PDF Services: extracting structured content from {pdf_basename}...")

    # ── Read PDF ──
    with open(pdf_path, "rb") as f:
        input_stream = f.read()

    # ── Authenticate & upload ──
    credentials = ServicePrincipalCredentials(
        client_id=os.getenv("PDF_SERVICES_CLIENT_ID"),
        client_secret=os.getenv("PDF_SERVICES_CLIENT_SECRET"),
    )
    pdf_services = PDFServices(credentials=credentials)
    input_asset = pdf_services.upload(
        input_stream=input_stream,
        mime_type=PDFServicesMediaType.PDF.mime_type,
    )

    # ── Configure extraction: TEXT + TABLES + FIGURE RENDITIONS ──
    extract_params = ExtractPDFParams(
        elements_to_extract=[
            ExtractElementType.TEXT,
            ExtractElementType.TABLES,
        ],
        # CRITICAL: Use ExtractRenditionsElementType (NOT ExtractElementType)
        # This is a DIFFERENT enum that includes FIGURES
        elements_to_extract_renditions=[
            ExtractRenditionsElementType.TABLES,
            ExtractRenditionsElementType.FIGURES,
        ],
    )

    # ── Submit job ──
    extract_job = ExtractPDFJob(
        input_asset=input_asset,
        extract_pdf_params=extract_params,
    )
    location = pdf_services.submit(extract_job)
    response = pdf_services.get_job_result(location, ExtractPDFResult)
    result = response.get_result()

    # ── Retrieve structured JSON ──
    content_json = result.get_content_json()
    if isinstance(content_json, bytes):
        content_json = json.loads(content_json)

    # ── Retrieve rendition ZIP (figures / table images) ──
    os.makedirs(FIGURES_DIR, exist_ok=True)
    saved_images = {}

    resource_asset = result.get_resource()          # Asset for the ZIP
    if resource_asset is not None:
        try:
            stream_asset = pdf_services.get_content(resource_asset)
            saved_images = _extract_renditions_from_zip(stream_asset)
        except Exception as e:
            print(f"  Warning: Could not retrieve renditions ZIP: {e}")

    # ── Parse JSON into chunks ──
    chunks = _parse_extract_json(content_json, saved_images)

    t = sum(1 for c in chunks if c["type"] == "text")
    tb = sum(1 for c in chunks if c["type"] == "table")
    fg = sum(1 for c in chunks if c["type"] == "figure")
    print(f"  -> Adobe extraction complete: {t} text, {tb} tables, {fg} figures")
    if saved_images:
        print(f"  -> Saved {len(saved_images)} rendition image(s) to {FIGURES_DIR}")

    return chunks


def prepare_multimodal_prompt(figure_chunk_metadata: dict, question: str) -> list:
    """
    Multimodal Integration Helper.

    Takes a retrieved figure chunk's metadata, reads the image file from
    image_path, converts it to Base64, and returns a content-parts list
    suitable for Gemini / GPT-4V / Claude multimodal prompts.

    Args:
        figure_chunk_metadata: The .metadata dict of a LangChain Document
            whose chunk_type == 'figure'.
        question: The user's question.

    Returns:
        List of content parts:
          [text_prompt, {"mime_type": ..., "data": base64_str}]
        Falls back to text-only if image is unavailable.
    """
    image_path = figure_chunk_metadata.get("image_path")
    figure_id = figure_chunk_metadata.get("figure_id", "Unknown Figure")
    page = figure_chunk_metadata.get("page", "?")

    text_prompt = (
        f"The following image is {figure_id} from page {page}. "
        "Analyze its visual structure (arrows, layers, labels, charts, diagrams) "
        f"to answer the question: {question}"
    )

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Detect MIME type
        ext = os.path.splitext(image_path)[1].lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
            ext.lstrip("."), "image/png"
        )

        return [text_prompt, {"mime_type": mime, "data": img_b64}]

    # Fallback: text-only
    return [text_prompt + " (Image file not available.)"]


# ═══════════════════════════════════════════════════════════════════════════════
# RENDITION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_renditions_from_zip(stream_asset) -> dict:
    """
    Extract image files from the Adobe renditions ZIP and save to assets/figures/.

    Returns:
        Dict mapping original ZIP entry path → local absolute path.
    """
    temp_dir = tempfile.mkdtemp()
    saved = {}

    try:
        zip_path = os.path.join(temp_dir, "renditions.zip")
        with open(zip_path, "wb") as f:
            f.write(stream_asset.get_input_stream())

        with zipfile.ZipFile(zip_path, "r") as zf:
            all_entries = zf.namelist()
            print(f"     ZIP contains {len(all_entries)} entries:")
            for entry in all_entries:
                print(f"       - {entry}")
            
            for entry in all_entries:
                lower = entry.lower()
                if lower.endswith((".png", ".jpg", ".jpeg")):
                    basename = os.path.basename(entry)
                    if not basename:
                        continue
                    dst = os.path.join(FIGURES_DIR, basename)
                    with zf.open(entry) as src, open(dst, "wb") as out:
                        out.write(src.read())
                    saved[entry] = dst
                    print(f"     Saved rendition: {entry} → {dst}")
    except Exception as e:
        print(f"  Warning: Failed to extract renditions: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"     Total saved images: {len(saved)}")
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# JSON PARSING
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_extract_json(content_json: dict, saved_images: dict) -> list:
    """
    Parse Adobe Extract API JSON into structured chunks.

    Walks every element, classifies by Path, and builds typed chunks.
    Figures get the Hybrid treatment (ID + caption + 500-char context).
    
    Uses Adobe's `filePaths` element key to match figures to their renditions.
    """
    elements = content_json.get("elements", [])
    chunks: list = []

    # Build a lookup: ZIP entry path → local path
    # saved_images = {"figures/fileoutpart0.png": "/abs/path/to/fileoutpart0.png", ...}
    # Adobe JSON elements have "filePaths": ["figures/fileoutpart0.png"]
    
    # Annotate each element with helpers
    for idx, el in enumerate(elements):
        el["_idx"] = idx
        el["_page"] = el.get("Page", 0) + 1          # 1-based
        el["_bounds"] = el.get("Bounds", [])

    fig_counter = 0
    table_counter = 0
    first_figure_printed = False
    
    for idx, el in enumerate(elements):
        path = el.get("Path", "")
        page = el["_page"]
        bounds = el["_bounds"]

        if "/Table" in path:
            is_child = "/TR" in path or "/TD" in path or "/TH" in path
            if not is_child:
                table_counter += 1
                chunk = _process_table_from_elements(
                    el, page, idx, table_counter, saved_images, elements,
                )
                if chunk:
                    chunks.append(chunk)
                    print(f"     [TABLE] p.{page}: Table[{table_counter}] ingested from element text")

        elif "/Figure" in path:
            fig_counter += 1
            
            # DEBUG: Print first figure element structure
            if not first_figure_printed:
                print(f"\n=== SAMPLE FIGURE ELEMENT FROM ADOBE JSON ===")
                print(f"Path: {el.get('Path')}")
                print(f"Page: {el.get('Page')}")
                print(f"Bounds: {el.get('Bounds')}")
                print(f"filePaths: {el.get('filePaths')}")
                print(f"Text: {el.get('Text', 'N/A')[:100]}")
                print(f"Keys in element: {list(el.keys())}")
                print(f"=============================================\n")
                first_figure_printed = True
            
            # ── Pydantic-validated figure pipeline ──
            # Extract caption and context first
            caption = _extract_caption(el, elements)
            ctx_before, ctx_after = _extract_context(el, elements, page, char_limit=500)
            image_path = _resolve_image_path(el, saved_images)
            
            from figure_models import validate_and_describe_figure
            chunk = validate_and_describe_figure(
                el=el,
                page=page,
                order=idx,
                all_elements=elements,
                saved_images=saved_images,
                fig_counter=fig_counter,
                caption=caption,
                context_before=ctx_before,
                context_after=ctx_after,
                image_path=image_path,
            )
            if chunk is not None:
                chunks.append(chunk)
            # Rejected figures are silently filtered out

        else:
            text = el.get("Text", "")
            if text and text.strip():
                is_heading = bool(re.search(r'/H\d?$|/H\d?[^a-zA-Z]', path))
                if is_heading:
                    print(f"     [HEADING] p.{page}: '{text.strip()[:60]}' (path={path})")
                chunks.append({
                    "type": "text",
                    "page": page,
                    "content": text.strip(),
                    "order": idx,
                    "bounds": bounds,
                    "is_heading": is_heading,
                })

    # Sort by reading order, then group paragraphs by section heading
    chunks = _sort_by_reading_order(chunks)
    chunks = _group_by_section(chunks)
    for i, c in enumerate(chunks):
        c["order"] = i

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE PROCESSING — Vision-Based (treat tables as images)
# ═══════════════════════════════════════════════════════════════════════════════


def _process_table_from_elements(
    el: dict,
    page: int,
    order: int,
    table_counter: int,
    saved_images: dict,
    all_elements: list,
) -> Optional[dict]:
    """
    Process a top-level table element by extracting text from its child
    elements (TD/TH cells) in the Adobe JSON. No Vision API call needed —
    on-demand Gemini Vision analysis at query time handles visual understanding.

    The rendition image (if available) is still saved and referenced in
    metadata for frontend display.
    """
    image_path = _resolve_image_path(el, saved_images)
    table_id = f"Table[{table_counter}]"
    table_path = el.get("Path", "")

    # Collect text from child TD/TH elements belonging to this table
    cell_texts = []
    for child in all_elements:
        child_path = child.get("Path", "")
        if not child_path.startswith(table_path):
            continue
        if "/TD" not in child_path and "/TH" not in child_path:
            continue
        text = (child.get("Text", "") or "").strip()
        if text:
            cell_texts.append(text)

    # Build text content from cells + surrounding context
    ctx_before, ctx_after = _extract_context(el, all_elements, page, char_limit=300)

    parts = [f"[{table_id} on page {page}]"]
    if cell_texts:
        parts.append(" | ".join(cell_texts))
    if ctx_before:
        parts.append(f"Context: {ctx_before[:200]}")
    if ctx_after:
        parts.append(ctx_after[:200])

    content = "\n".join(parts)

    if not content.strip() or (not cell_texts and not ctx_before):
        content = f"Table on page {page} (visual content — see attached image)"

    return {
        "type": "table",
        "page": page,
        "content": content,
        "order": order,
        "bounds": el.get("_bounds", []),
        "image_path": image_path if image_path else "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE PROCESSING — Hybrid Chunk (Semantic Context Anchoring)
# ═══════════════════════════════════════════════════════════════════════════════


def _process_figure(
    el: dict,
    page: int,
    order: int,
    all_elements: list,
    saved_images: dict,
    fig_counter: int,
) -> dict:
    """
    Build a HYBRID figure chunk that merges:
      [Figure ID] + [Adobe-extracted Caption] + [500 chars before & after]

    ZERO-NULL POLICY: Always returns a dict with non-empty content.

    Uses Adobe's `filePaths` key to directly match element → rendition image.
    """
    path = el.get("Path", "")
    figure_id = path.split("/")[-1] if "/" in path else f"Figure[{fig_counter}]"

    # ── Find image using Adobe's filePaths mapping ──
    image_path = _resolve_image_path(el, saved_images)
    if image_path:
        print(f"     Matched {figure_id} → {os.path.basename(image_path)}")
    else:
        print(f"     Warning: No rendition image found for {figure_id}")

    # ── Extract caption ──
    caption = _extract_caption(el, all_elements)

    # ── Extract 500-char context before & after ──
    ctx_before, ctx_after = _extract_context(el, all_elements, page, char_limit=500)

    # ── Build hybrid content ──
    parts: list = []

    if ctx_before:
        parts.append(ctx_before)

    parts.append(f"\n[FIGURE: {figure_id}]")

    if caption:
        parts.append(f"Caption: {caption}")

    if image_path:
        parts.append(f"(Image saved: {os.path.basename(image_path)})")

    if ctx_after:
        parts.append(ctx_after)

    # ZERO-NULL: guarantee content
    if len(parts) <= 2:
        parts.append(f"Visual element on page {page}.")
        inline_text = el.get("Text", "")
        if inline_text and inline_text.strip():
            parts.append(inline_text.strip())

    content = "\n".join(parts)

    return {
        "type": "figure",
        "page": page,
        "content": content,
        "order": order,
        "figure_id": figure_id,
        # Use empty string instead of None — ChromaDB cannot store None values
        "image_path": image_path if image_path else "",
        "bounds": el.get("_bounds", []),
    }


def _resolve_image_path(el: dict, saved_images: dict) -> Optional[str]:
    """
    Match a figure element to its saved rendition image.
    
    Uses Adobe's `filePaths` key which directly maps elements to rendition files.
    Adobe JSON example: {"Path": "//Document/Figure", "filePaths": ["figures/fileoutpart0.png"]}
    saved_images maps ZIP entry paths to local absolute paths.
    """
    if not saved_images:
        return None

    # PRIMARY: Use Adobe's filePaths key (direct mapping)
    file_paths = el.get("filePaths", [])
    for fp in file_paths:
        # fp looks like "figures/fileoutpart0.png" or "fileoutpart0.png"
        if fp in saved_images:
            return saved_images[fp]
        # Also try matching by basename
        base = os.path.basename(fp)
        for zip_name, local_path in saved_images.items():
            if os.path.basename(zip_name) == base:
                return local_path

    # FALLBACK: Check if element's file path ref matches any saved image
    file_ref = el.get("File", "") or el.get("fileRef", "")
    if file_ref:
        for zip_name, local_path in saved_images.items():
            if file_ref in zip_name or os.path.basename(file_ref) == os.path.basename(zip_name):
                return local_path

    return None


def _extract_caption(el: dict, all_elements: list) -> str:
    """Extract figure caption from Title, Alt, or child Caption elements."""
    caption = el.get("Title", "") or el.get("Alt", "")
    if caption:
        return caption.strip()

    fig_path = el.get("Path", "")
    for other in all_elements:
        other_path = other.get("Path", "")
        if other_path.startswith(fig_path) and "/Caption" in other_path:
            return (other.get("Text", "") or "").strip()

    return ""


def _extract_context(
    el: dict, all_elements: list, page: int, char_limit: int = 500
) -> Tuple[str, str]:
    """
    Gather surrounding text (up to char_limit chars) before and after a figure
    on the same page, using spatial (Y-coordinate) proximity.
    """
    bounds = el.get("_bounds", [])
    if not bounds or len(bounds) < 4:
        return "", ""

    fig_y_top = bounds[1]
    fig_y_bot = bounds[3]

    before: list = []
    after: list = []

    for other in all_elements:
        if other.get("_page", 0) != page:
            continue
        opath = other.get("Path", "")
        if "/P" not in opath and "/H" not in opath:
            continue
        ob = other.get("_bounds", [])
        if not ob or len(ob) < 4:
            continue
        text = (other.get("Text", "") or "").strip()
        if not text:
            continue

        oy = ob[1]
        if oy < fig_y_top:
            before.append((oy, text))
        elif oy > fig_y_bot:
            after.append((oy, text))

    # Sort: closest to figure first
    before.sort(key=lambda x: x[0], reverse=True)
    after.sort(key=lambda x: x[0])

    # Accumulate up to char_limit
    ctx_before = _accumulate_context(before, char_limit)
    ctx_after = _accumulate_context(after, char_limit)

    return ctx_before, ctx_after


def _accumulate_context(sorted_items: list, char_limit: int) -> str:
    """Join text items until char_limit is reached."""
    parts: list = []
    total = 0
    for _, text in sorted_items:
        if total + len(text) > char_limit:
            remaining = char_limit - total
            if remaining > 50:      # only add if meaningful
                parts.append(text[:remaining] + "…")
            break
        parts.append(text)
        total += len(text) + 1  # +1 for space

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# SORTING
# ═══════════════════════════════════════════════════════════════════════════════


def _sort_by_reading_order(chunks: list) -> list:
    """Sort by page → Y (top-to-bottom) → X (left-to-right)."""

    def key(c):
        b = c.get("bounds", [])
        if not b or len(b) < 4:
            return (c.get("page", 0), 0, c.get("order", 0))
        return (c.get("page", 0), -b[1], b[0])

    return sorted(chunks, key=key)


def _group_by_section(chunks: list) -> list:
    """
    Group consecutive text paragraphs under the same section heading,
    producing larger, coherent section chunks for academic papers.

    Walks all chunks in reading order, tracking the current heading.
    Consecutive body paragraphs under the same heading are concatenated
    into a single section chunk. Tables and figures pass through unchanged
    with `section_heading` attached from the most recent heading.

    Each output chunk gets a `section_heading` field (str or "").
    """
    result = []
    current_heading = ""
    section_buffer = None  # Accumulates text for the current section

    def _flush_section():
        """Append the buffered section chunk to results."""
        nonlocal section_buffer
        if section_buffer:
            result.append(section_buffer)
            section_buffer = None

    for chunk in chunks:
        # Tables and figures: flush any pending section, pass through
        if chunk["type"] != "text":
            _flush_section()
            chunk["section_heading"] = current_heading
            result.append(chunk)
            continue

        # Heading: starts a new section
        if chunk.get("is_heading"):
            _flush_section()
            current_heading = chunk["content"]
            print(f"  [SECTION] p.{chunk['page']}: '{current_heading[:60]}'")
            # Start new section buffer with heading as prefix
            section_buffer = {
                "type": "text",
                "page": chunk["page"],
                "content": current_heading,
                "order": chunk["order"],
                "bounds": chunk["bounds"],
                "is_heading": False,
                "section_heading": current_heading,
            }
            continue

        # Body paragraph: append to current section or start standalone
        if section_buffer and chunk["page"] == section_buffer["page"]:
            # Same page — extend the section
            section_buffer["content"] += "\n" + chunk["content"]
            # Expand bounds to cover both chunks
            if chunk.get("bounds") and section_buffer.get("bounds"):
                sb = section_buffer["bounds"]
                cb = chunk["bounds"]
                if len(sb) >= 4 and len(cb) >= 4:
                    section_buffer["bounds"] = [
                        min(sb[0], cb[0]),
                        min(sb[1], cb[1]),
                        max(sb[2], cb[2]),
                        max(sb[3], cb[3]),
                    ]
        elif section_buffer:
            # Different page — flush old section, start new one
            _flush_section()
            section_buffer = {
                "type": "text",
                "page": chunk["page"],
                "content": chunk["content"],
                "order": chunk["order"],
                "bounds": chunk["bounds"],
                "is_heading": False,
                "section_heading": current_heading,
            }
        else:
            # No pending section — start one
            section_buffer = {
                "type": "text",
                "page": chunk["page"],
                "content": chunk["content"],
                "order": chunk["order"],
                "bounds": chunk["bounds"],
                "is_heading": False,
                "section_heading": current_heading,
            }

    _flush_section()
    return result
