# PDF Reference Highlighting — Design Document

**Date**: 2026-03-30
**Feature**: Highlight RAG source references directly on the PDF document

## Goal

When the RAG answers a question and cites sources, the user can see exactly where on the original PDF those references come from, with the relevant regions highlighted in an embedded PDF viewer.

## User Experience

- A persistent PDF viewer panel sits to the right of the chat.
- The panel is collapsible (toggle button in UI).
- When a user clicks an ingested PDF document or a citation in a chat response, the PDF viewer loads that document and scrolls to the relevant page.
- All retrieved source regions for the current answer are shown as semi-transparent colored overlays on the PDF pages.
- All highlights show dimly; the clicked/focused citation gets a prominent highlight.
- Each citation gets a unique color for visual distinction.

## Architecture Changes

### Backend

1. **`adobe_ocr.py`** — Stop stripping bounding boxes from chunks. Keep `bounds` in the chunk dict.
2. **`ingest_single_file.py`** — Store bounds as individual ChromaDB metadata fields (`bounds_x_min`, `bounds_y_min`, `bounds_x_max`, `bounds_y_max`). ChromaDB doesn't support list values in metadata.
3. **`rag_chatbot.py`**:
   - `build_sources()` includes bounds in the JSON response.
   - New endpoint: `GET /api/documents/<filename>/file` to serve original PDFs.
4. Re-ingestion required for existing documents (bounds weren't stored before).

### Frontend

1. **Dependency**: `react-pdf` (wraps PDF.js) for rendering PDF pages.
2. **PdfViewer component**: Renders PDF pages with navigation, accepts highlight data.
3. **Layout**: `[sidebar] [chat] [PDF viewer]` — three-column when viewer is open.
4. **Highlight overlays**: Canvas or absolute-positioned divs over PDF pages at bounding box positions.
5. **Click-to-jump**: Clicking a citation in chat scrolls the PDF viewer to that page/region.

### API Response Shape (Updated)

```json
{
  "sources": [
    {
      "id": 1,
      "source": "paper.pdf",
      "page": 3,
      "bounds": { "x_min": 72.0, "y_min": 400.0, "x_max": 540.0, "y_max": 520.0 },
      "chunk_type": "text",
      "preview": "...",
      ...
    }
  ]
}
```

## Coordinate System

Adobe PDF Extract uses PDF points with origin at bottom-left, Y increasing upward. PDF.js uses CSS pixels with origin at top-left, Y increasing downward. A coordinate transform is needed when rendering highlights:

```
css_x = pdf_x * scale
css_y = (page_height - pdf_y) * scale
```

## Constraints

- No new `package.json` files.
- All config via env variables.
- CSS in `.css` files only.
- Traditional Chinese for Chinese text, English for code comments.
