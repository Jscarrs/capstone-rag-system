"""
Pydantic Figure Models — Validation, Filtering & Description

This module provides structured validation for figure elements extracted
by Adobe PDF Services. It filters out false positives (decorative elements,
table fragments, tiny icons) and generates rich text descriptions using
spatial reasoning (zero-API) or optionally Gemini Vision.

Key Design Decisions:
  - Pydantic models enforce structured, typed figure data
  - Quality scoring filters out unusable figure fragments
  - ENABLE_VISION_INGESTION flag controls whether Gemini Vision runs at ingestion
  - Spatial synthesis provides a robust zero-API fallback for descriptions
  - Lazy Vision at query time handles multimodal answers on-demand
  - Source tracking (page, figure_id, document) preserved for citations
"""

import os
import base64
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class FigureType(str, Enum):
    """Classification of figure elements from Adobe PDF extraction."""
    DIAGRAM = "diagram"
    CHART = "chart"
    PHOTOGRAPH = "photograph"
    ILLUSTRATION = "illustration"
    ICON = "icon"
    DECORATIVE = "decorative"
    TABLE_FRAGMENT = "table_fragment"
    UNKNOWN = "unknown"


class FigureQuality(str, Enum):
    """Quality assessment result."""
    HIGH = "high"         # Full figure with caption, good size
    MEDIUM = "medium"     # Usable figure, may lack caption
    LOW = "low"           # Small or missing context — borderline
    REJECTED = "rejected" # False positive — filtered out


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class FigureBounds(BaseModel):
    """Spatial bounds of a figure on the PDF page (Adobe coordinate system)."""
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0

    @property
    def width(self) -> float:
        return abs(self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return abs(self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_adobe_bounds(cls, bounds: list) -> "FigureBounds":
        """Parse Adobe's [x_min, y_min, x_max, y_max] format."""
        if not bounds or len(bounds) < 4:
            return cls()
        return cls(
            x_min=float(bounds[0]),
            y_min=float(bounds[1]),
            x_max=float(bounds[2]),
            y_max=float(bounds[3]),
        )


class FigureCandidate(BaseModel):
    """
    Raw figure element from Adobe PDF extraction — before validation.
    
    This is the input model that captures everything Adobe gives us
    about a figure element, before we decide if it's worth keeping.
    """
    figure_id: str = Field(description="Figure identifier from Adobe Path")
    page: int = Field(ge=1, description="1-based page number")
    order: int = Field(ge=0, description="Reading order index")
    bounds: FigureBounds = Field(default_factory=FigureBounds)
    
    # Content from Adobe
    caption: str = Field(default="", description="Extracted caption text")
    context_before: str = Field(default="", description="Text before the figure")
    context_after: str = Field(default="", description="Text after the figure")
    inline_text: str = Field(default="", description="Any text inside the figure element")
    
    # Image info
    image_path: str = Field(default="", description="Path to saved rendition image")
    has_image: bool = Field(default=False, description="Whether a rendition image exists")
    image_size_bytes: int = Field(default=0, description="File size of the image")
    
    # Adobe metadata
    adobe_path: str = Field(default="", description="Full Adobe element Path")
    file_paths: List[str] = Field(default_factory=list, description="Adobe filePaths")

    @field_validator("caption", "context_before", "context_after", "inline_text", mode="before")
    @classmethod
    def clean_text(cls, v):
        """Strip and normalize whitespace."""
        if not v:
            return ""
        return str(v).strip()

    @property
    def total_context_length(self) -> int:
        """Total characters of contextual text available."""
        return len(self.caption) + len(self.context_before) + len(self.context_after) + len(self.inline_text)


class ValidatedFigure(BaseModel):
    """
    A validated, quality-checked figure ready for ingestion.
    
    Only figures that pass quality checks become ValidatedFigures.
    Contains either a Gemini-generated description or a synthesized
    text description for storage in ChromaDB.
    """
    figure_id: str
    page: int
    order: int
    source_document: str = Field(default="", description="Source PDF filename")
    
    # Quality assessment
    quality: FigureQuality
    quality_score: float = Field(ge=0.0, le=1.0, description="0.0-1.0 quality score")
    figure_type: FigureType = FigureType.UNKNOWN
    rejection_reason: str = Field(default="", description="Why this figure was rejected")
    
    # Content — the rich text description
    description: str = Field(description="Rich text description of the figure")
    caption: str = Field(default="")
    context_before: str = Field(default="")
    context_after: str = Field(default="")
    
    # Image reference (kept for potential future use, but NOT sent to LLM)
    image_path: str = Field(default="")
    
    # Spatial bounds (needed for spatial-reasoning description synthesis)
    bounds: Optional[FigureBounds] = Field(default=None, description="Figure's spatial bounds")
    
    @property
    def chunk_content(self) -> str:
        """
        Build the final chunk content for ChromaDB storage.
        Combines all textual information into a searchable chunk.
        """
        parts = []

        if self.context_before:
            parts.append(self.context_before)

        parts.append(f"\n[FIGURE: {self.figure_id}]")
        parts.append(f"Type: {self.figure_type.value}")

        if self.caption:
            parts.append(f"Caption: {self.caption}")

        if self.description:
            parts.append(f"Description: {self.description}")

        if self.context_after:
            parts.append(self.context_after)

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY ASSESSMENT & FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

# Thresholds for filtering
MIN_BOUNDS_AREA = 2000.0        # Minimum area in PDF points² (filters tiny icons)
MIN_BOUNDS_WIDTH = 30.0         # Minimum width in PDF points
MIN_BOUNDS_HEIGHT = 30.0        # Minimum height in PDF points
MIN_IMAGE_SIZE_BYTES = 1024     # Minimum 1KB image file (filters blank/tiny)
MIN_QUALITY_SCORE = 0.3         # Minimum score to keep a figure


def assess_figure_quality(candidate: FigureCandidate) -> ValidatedFigure:
    """
    Assess quality of a figure candidate and decide whether to keep or reject.
    
    Scoring system (0.0 - 1.0):
      +0.25 — Has meaningful caption
      +0.20 — Has surrounding context (before or after)
      +0.20 — Has rendition image of decent size
      +0.15 — Bounds area exceeds minimum
      +0.10 — Has inline text
      +0.10 — Has Adobe filePaths (proper figure, not fragment)
    
    Automatic rejection (score = 0):
      - Bounds area < MIN_BOUNDS_AREA (tiny decorative element)
      - Image size < MIN_IMAGE_SIZE_BYTES (blank placeholder)
      - No image AND no caption AND no context (nothing usable)
    """
    score = 0.0
    rejection_reason = ""
    fig_type = FigureType.UNKNOWN

    # ── Automatic rejection checks ──
    
    # Check 1: Bounds too small (icon/decorative)
    if candidate.bounds.area > 0 and candidate.bounds.area < MIN_BOUNDS_AREA:
        return ValidatedFigure(
            figure_id=candidate.figure_id,
            page=candidate.page,
            order=candidate.order,
            quality=FigureQuality.REJECTED,
            quality_score=0.0,
            figure_type=FigureType.DECORATIVE,
            rejection_reason=f"Too small: area={candidate.bounds.area:.0f} < {MIN_BOUNDS_AREA}",
            description="",
            caption=candidate.caption,
            image_path=candidate.image_path,
            bounds=candidate.bounds,
        )

    # Check 2: Dimensions too narrow (table border/line)
    if candidate.bounds.width > 0 and candidate.bounds.width < MIN_BOUNDS_WIDTH:
        return ValidatedFigure(
            figure_id=candidate.figure_id,
            page=candidate.page,
            order=candidate.order,
            quality=FigureQuality.REJECTED,
            quality_score=0.0,
            figure_type=FigureType.TABLE_FRAGMENT,
            rejection_reason=f"Too narrow: width={candidate.bounds.width:.0f} < {MIN_BOUNDS_WIDTH}",
            description="",
            caption=candidate.caption,
            image_path=candidate.image_path,
            bounds=candidate.bounds,
        )

    if candidate.bounds.height > 0 and candidate.bounds.height < MIN_BOUNDS_HEIGHT:
        return ValidatedFigure(
            figure_id=candidate.figure_id,
            page=candidate.page,
            order=candidate.order,
            quality=FigureQuality.REJECTED,
            quality_score=0.0,
            figure_type=FigureType.TABLE_FRAGMENT,
            rejection_reason=f"Too short: height={candidate.bounds.height:.0f} < {MIN_BOUNDS_HEIGHT}",
            description="",
            caption=candidate.caption,
            image_path=candidate.image_path,
            bounds=candidate.bounds,
        )

    # Check 3: No usable content at all
    if not candidate.has_image and candidate.total_context_length < 20:
        return ValidatedFigure(
            figure_id=candidate.figure_id,
            page=candidate.page,
            order=candidate.order,
            quality=FigureQuality.REJECTED,
            quality_score=0.0,
            figure_type=FigureType.DECORATIVE,
            rejection_reason="No image and insufficient context text",
            description="",
            caption=candidate.caption,
            image_path=candidate.image_path,
            bounds=candidate.bounds,
        )

    # Check 4: Image file too small (blank/placeholder)
    if candidate.has_image and candidate.image_size_bytes < MIN_IMAGE_SIZE_BYTES:
        score -= 0.1  # Penalty but not auto-reject (might have good context)

    # ── Positive scoring ──
    
    # Caption quality (+0.25)
    if candidate.caption and len(candidate.caption) > 10:
        score += 0.25
    elif candidate.caption:
        score += 0.10

    # Context quality (+0.20)
    context_len = len(candidate.context_before) + len(candidate.context_after)
    if context_len > 200:
        score += 0.20
    elif context_len > 50:
        score += 0.10

    # Image quality (+0.20)
    if candidate.has_image and candidate.image_size_bytes > 5000:
        score += 0.20
    elif candidate.has_image:
        score += 0.10

    # Bounds size (+0.15)
    if candidate.bounds.area > 10000:
        score += 0.15
    elif candidate.bounds.area > 5000:
        score += 0.08

    # Inline text (+0.10)
    if candidate.inline_text and len(candidate.inline_text) > 10:
        score += 0.10

    # Adobe filePaths present (+0.10)
    if candidate.file_paths:
        score += 0.10

    # ── Classify figure type based on heuristics ──
    caption_lower = (candidate.caption + " " + candidate.context_before + " " + candidate.context_after).lower()
    if any(w in caption_lower for w in ["chart", "graph", "plot", "histogram", "bar chart", "pie"]):
        fig_type = FigureType.CHART
    elif any(w in caption_lower for w in ["diagram", "architecture", "flow", "pipeline", "workflow", "schema"]):
        fig_type = FigureType.DIAGRAM
    elif any(w in caption_lower for w in ["photo", "image", "picture", "screenshot"]):
        fig_type = FigureType.PHOTOGRAPH
    elif any(w in caption_lower for w in ["illustration", "drawing", "sketch"]):
        fig_type = FigureType.ILLUSTRATION

    # Clamp score
    score = max(0.0, min(1.0, score))

    # Determine quality tier
    if score >= 0.5:
        quality = FigureQuality.HIGH
    elif score >= MIN_QUALITY_SCORE:
        quality = FigureQuality.MEDIUM
    else:
        quality = FigureQuality.LOW
        rejection_reason = f"Low quality score: {score:.2f} < {MIN_QUALITY_SCORE}"

    return ValidatedFigure(
        figure_id=candidate.figure_id,
        page=candidate.page,
        order=candidate.order,
        quality=quality,
        quality_score=score,
        figure_type=fig_type,
        rejection_reason=rejection_reason,
        description="",  # Will be filled by describe_figure()
        caption=candidate.caption,
        context_before=candidate.context_before,
        context_after=candidate.context_after,
        image_path=candidate.image_path,
        bounds=candidate.bounds,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI VISION DESCRIPTION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def describe_figure_with_vision(
    validated: ValidatedFigure,
    all_elements: list,
) -> str:
    """
    Generate a text description for a validated figure.
    
    Strategy is controlled by ENABLE_VISION_INGESTION env var:
      True  → Try Gemini Vision API, fall back to spatial synthesis
      False → Skip API entirely, use spatial synthesis only (default)
    
    This flag is SEPARATE from GOOGLE_API_KEY — the key stays active
    for the chatbot's lazy Vision at query time.
    """
    # Check explicit feature flag (default: OFF to avoid 429 during ingestion)
    vision_enabled = os.getenv("ENABLE_VISION_INGESTION", "false").lower() == "true"

    if vision_enabled:
        # Try Gemini Vision if image exists AND flag is on
        if validated.image_path and os.path.exists(validated.image_path):
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if api_key:
                try:
                    description = _call_gemini_vision(validated.image_path, validated)
                    if description:
                        print(f"       Vision description generated (ingestion-time)")
                        return description
                except Exception as e:
                    print(f"     Warning: Gemini Vision failed for {validated.figure_id}: {e}")
    else:
        print(f"       ENABLE_VISION_INGESTION=false, using spatial synthesis only")

    # Fallback: spatial-reasoning text synthesis (zero API calls)
    return _synthesize_text_description(validated, all_elements)


def _call_gemini_vision(image_path: str, validated: ValidatedFigure) -> str:
    """Call Gemini Vision API to describe a figure image."""
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY", "")
    client = genai.Client(api_key=api_key)

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")

    # Build context-aware prompt
    context_parts = []
    if validated.caption:
        context_parts.append(f"Caption: {validated.caption}")
    if validated.context_before:
        context_parts.append(f"Preceding text: {validated.context_before[:200]}")
    if validated.context_after:
        context_parts.append(f"Following text: {validated.context_after[:200]}")

    context_str = "\n".join(context_parts) if context_parts else "No additional context."

    prompt = (
        "You are analyzing a figure from an academic/technical PDF document. "
        "Describe this figure in detail for a text-based search and retrieval system. "
        "Your description must be comprehensive enough that someone can understand "
        "the figure WITHOUT seeing it.\n\n"
        "Include:\n"
        "1. WHAT the figure shows (chart type, diagram type, photo subject, etc.)\n"
        "2. KEY LABELS and text visible in the figure\n"
        "3. DATA or VALUES shown (numbers, percentages, measurements)\n"
        "4. RELATIONSHIPS or FLOWS depicted (arrows, connections, hierarchies)\n"
        "5. NOTABLE PATTERNS or TRENDS visible\n"
        "6. COLOR CODING or LEGEND information if present\n\n"
        f"Context from the document:\n{context_str}\n\n"
        "Write a thorough, factual description in 3-8 sentences. "
        "Do NOT start with 'This figure shows' — be direct and specific."
    )

    # Upload image and generate description
    image_part = genai.types.Part.from_bytes(data=img_bytes, mime_type=mime)
    response = client.models.generate_content(
        model=os.getenv("VISION_MODEL_NAME", "gemini-3.1-flash-lite-preview"),
        contents=[prompt, image_part],
    )

    return response.text.strip()


def _synthesize_text_description(
    validated: ValidatedFigure,
    all_elements: list,
) -> str:
    """
    Spatial-Reasoning Text Synthesis — 100% API-free.

    Instead of just dumping caption + surrounding text, this function uses the
    figure's bounding box to find every Adobe element that is INSIDE the figure.
    These internal snippets (flowchart box labels, axis titles, legend items,
    etc.) are sorted top-to-bottom by Y-coordinate to reconstruct the visual
    reading order.

    The final description combines:
      1. Figure type + page location
      2. Caption (if any)
      3. Internal text sequence (spatial scan)
      4. Context before/after the figure
    """
    parts = []

    # ── Header ──
    fig_type_label = validated.figure_type.value.replace("_", " ").title()
    parts.append(f"{fig_type_label} on page {validated.page}.")

    if validated.caption:
        parts.append(f'Caption: "{validated.caption}"')

    # ── Spatial scan: extract text INSIDE the figure bounds ──
    internal_snippets = _extract_internal_text(
        all_elements=all_elements,
        page=validated.page,
        bounds=validated.bounds if hasattr(validated, 'bounds') else None,
    )

    if internal_snippets:
        parts.append(f"Internal text elements ({len(internal_snippets)} found, top-to-bottom):")
        for snippet in internal_snippets:
            parts.append(f"  • {snippet}")
    else:
        parts.append("No internal text elements detected within figure bounds.")

    # ── Surrounding context ──
    if validated.context_before:
        ctx = validated.context_before[:300]
        parts.append(f"Preceding context: {ctx}")

    if validated.context_after:
        ctx = validated.context_after[:300]
        parts.append(f"Following context: {ctx}")

    if (not validated.caption
            and not internal_snippets
            and not validated.context_before
            and not validated.context_after):
        parts.append("No textual information available for this visual element.")

    return "\n".join(parts)


def _extract_internal_text(
    all_elements: list,
    page: int,
    bounds: Optional["FigureBounds"] = None,
) -> List[str]:
    """
    Scan all_elements for text snippets located INSIDE the figure's bounding box.

    Uses Adobe's spatial coordinates (_bounds, _page) to find elements whose
    center point falls within the figure. Sorts top-to-bottom by Y-coordinate
    to simulate the visual reading order (e.g., flowchart boxes in sequence).

    Returns:
        List of text strings, ordered top-to-bottom.
    """
    if not all_elements or bounds is None:
        return []

    # Figure bounding box (Adobe coordinate system: origin at bottom-left,
    # Y increases upward on the page)
    fig_x_min = bounds.x_min
    fig_y_min = bounds.y_min
    fig_x_max = bounds.x_max
    fig_y_max = bounds.y_max

    # Skip if bounds are zero/invalid
    if fig_x_max <= fig_x_min or fig_y_max <= fig_y_min:
        return []

    hits: list = []  # (y_center, text)

    for el in all_elements:
        # Must be on the same page
        if el.get("_page", 0) != page:
            continue

        # Skip the figure element itself and other figures
        el_path = el.get("Path", "")
        if "/Figure" in el_path:
            continue

        # Must have text content
        text = (el.get("Text", "") or "").strip()
        if not text or len(text) < 2:
            continue

        # Check if this element's center lies inside the figure bounds
        el_bounds = el.get("_bounds", [])
        if not el_bounds or len(el_bounds) < 4:
            continue

        el_x_min, el_y_min, el_x_max, el_y_max = (
            float(el_bounds[0]), float(el_bounds[1]),
            float(el_bounds[2]), float(el_bounds[3]),
        )

        # Center point of the element
        cx = (el_x_min + el_x_max) / 2
        cy = (el_y_min + el_y_max) / 2

        # Is the center inside the figure's bounding box?
        if fig_x_min <= cx <= fig_x_max and fig_y_min <= cy <= fig_y_max:
            hits.append((cy, text))

    # Sort top-to-bottom (in Adobe coords, higher Y = higher on page)
    hits.sort(key=lambda h: -h[0])

    # Deduplicate consecutive identical texts
    result: list = []
    prev = None
    for _, text in hits:
        if text != prev:
            result.append(text)
            prev = text

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTION — Used by adobe_ocr.py
# ═══════════════════════════════════════════════════════════════════════════════

def validate_and_describe_figure(
    el: dict,
    page: int,
    order: int,
    all_elements: list,
    saved_images: dict,
    fig_counter: int,
    caption: str,
    context_before: str,
    context_after: str,
    image_path: Optional[str],
) -> Optional[dict]:
    """
    Full pipeline: validate → filter → describe → return chunk dict.
    
    This replaces the old _process_figure() approach.
    Returns None for rejected figures (false positives).
    Returns a structured chunk dict for valid figures.
    """
    # Get image size
    image_size = 0
    if image_path and os.path.exists(image_path):
        image_size = os.path.getsize(image_path)

    # Build Pydantic candidate
    path = el.get("Path", "")
    figure_id = path.split("/")[-1] if "/" in path else f"Figure[{fig_counter}]"

    candidate = FigureCandidate(
        figure_id=figure_id,
        page=page,
        order=order,
        bounds=FigureBounds.from_adobe_bounds(el.get("_bounds", [])),
        caption=caption,
        context_before=context_before,
        context_after=context_after,
        inline_text=el.get("Text", "") or "",
        image_path=image_path or "",
        has_image=bool(image_path and os.path.exists(str(image_path))),
        image_size_bytes=image_size,
        adobe_path=path,
        file_paths=el.get("filePaths", []),
    )

    # Validate quality
    validated = assess_figure_quality(candidate)

    # Reject false positives
    if validated.quality == FigureQuality.REJECTED:
        print(f"     REJECTED {figure_id}: {validated.rejection_reason}")
        return None

    # Generate description for valid figures
    print(f"     ACCEPTED {figure_id} (quality={validated.quality.value}, score={validated.quality_score:.2f}, type={validated.figure_type.value})")
    validated.description = describe_figure_with_vision(validated, all_elements)

    if validated.description:
        desc_preview = validated.description[:80].replace("\n", " ")
        print(f"       Description: {desc_preview}...")

    # Build bounds list from Pydantic model
    bounds = []
    if validated.bounds:
        bounds = [
            validated.bounds.x_min, validated.bounds.y_min,
            validated.bounds.x_max, validated.bounds.y_max,
        ]

    # Return chunk dict (compatible with existing pipeline)
    return {
        "type": "figure",
        "page": page,
        "content": validated.chunk_content,
        "order": order,
        "figure_id": validated.figure_id,
        "image_path": validated.image_path or "",
        "figure_type": validated.figure_type.value,
        "quality_score": validated.quality_score,
        "description": validated.description,
        "bounds": bounds,
    }
