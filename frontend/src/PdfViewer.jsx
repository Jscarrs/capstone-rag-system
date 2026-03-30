/**
 * PdfViewer — Embedded PDF viewer with highlight overlays
 *
 * Requirements:
 * - Render PDF pages using react-pdf (PDF.js wrapper)
 * - Overlay bounding-box highlights from RAG source citations
 * - Support click-to-jump: scroll to specific page + highlight a focused source
 * - Show all source highlights dimly; the focused one is prominent
 * - Each citation gets a unique color
 * - Page navigation (prev/next + page number input)
 * - Coordinate transform: Adobe PDF coords (bottom-left origin) -> CSS (top-left origin)
 *
 * Props:
 *   pdfUrl        - URL to fetch the PDF file
 *   sources       - Array of source objects with { id, page, bounds, source }
 *   focusedSource - The source id currently focused (clicked citation), or null
 *   onClose       - Callback to close/collapse the viewer
 *   onHighlightClick - Callback when a highlight overlay is clicked (source id)
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";
import { ChevronLeft, ChevronRight, X, Minus, Plus } from "lucide-react";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

const HIGHLIGHT_COLORS = [
  "rgba(59,130,246,0.25)",
  "rgba(16,185,129,0.25)",
  "rgba(245,158,11,0.25)",
  "rgba(239,68,68,0.25)",
  "rgba(139,92,246,0.25)",
  "rgba(236,72,153,0.25)",
  "rgba(6,182,212,0.25)",
  "rgba(132,204,22,0.25)",
];

const FOCUSED_COLORS = [
  "rgba(59,130,246,0.50)",
  "rgba(16,185,129,0.50)",
  "rgba(245,158,11,0.50)",
  "rgba(239,68,68,0.50)",
  "rgba(139,92,246,0.50)",
  "rgba(236,72,153,0.50)",
  "rgba(6,182,212,0.50)",
  "rgba(132,204,22,0.50)",
];

const BORDER_COLORS = [
  "rgba(59,130,246,0.80)",
  "rgba(16,185,129,0.80)",
  "rgba(245,158,11,0.80)",
  "rgba(239,68,68,0.80)",
  "rgba(139,92,246,0.80)",
  "rgba(236,72,153,0.80)",
  "rgba(6,182,212,0.80)",
  "rgba(132,204,22,0.80)",
];

export default function PdfViewer({
  pdfUrl,
  sources,
  focusedSource,
  onClose,
  onHighlightClick,
}) {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [pageDimensions, setPageDimensions] = useState(null);
  const [loading, setLoading] = useState(true);
  const containerRef = useRef(null);
  const pageRef = useRef(null);

  const onDocumentLoadSuccess = useCallback(({ numPages: total }) => {
    setNumPages(total);
    setLoading(false);
    console.log("[PdfViewer] loaded PDF with", total, "pages");
  }, []);

  const onPageLoadSuccess = useCallback((page) => {
    setPageDimensions({
      width: page.originalWidth,
      height: page.originalHeight,
    });
  }, []);

  useEffect(() => {
    if (focusedSource != null && sources) {
      const src = sources.find((s) => s.id === focusedSource);
      if (src?.page) {
        setPageNumber(src.page);
      }
    }
  }, [focusedSource, sources]);

  const currentPageSources = useMemo(() => {
    if (!sources) return [];
    return sources.filter((s) => s.page === pageNumber && s.bounds);
  }, [sources, pageNumber]);

  const allPagesWithSources = useMemo(() => {
    if (!sources) return new Set();
    return new Set(sources.filter((s) => s.bounds).map((s) => s.page));
  }, [sources]);

  function goToPage(p) {
    const target = Math.max(1, Math.min(p, numPages || 1));
    setPageNumber(target);
  }

  function adjustScale(delta) {
    setScale((prev) => Math.max(0.5, Math.min(3.0, prev + delta)));
  }

  function transformBounds(bounds, dims, currentScale) {
    if (!bounds || !dims) return null;
    const sx = currentScale;
    const left = bounds.x_min * sx;
    const right = bounds.x_max * sx;
    const top = (dims.height - bounds.y_max) * sx;
    const bottom = (dims.height - bounds.y_min) * sx;
    return {
      left,
      top,
      width: right - left,
      height: bottom - top,
    };
  }

  if (!pdfUrl) return null;

  return (
    <div className="pdf-viewer-panel">
      <div className="pdf-viewer-toolbar">
        <div className="pdf-viewer-nav">
          <button
            className="pdf-nav-btn"
            disabled={pageNumber <= 1}
            onClick={() => goToPage(pageNumber - 1)}
            title="Previous page"
          >
            <ChevronLeft size={16} />
          </button>
          <span className="pdf-page-info">
            <input
              type="number"
              className="pdf-page-input"
              value={pageNumber}
              min={1}
              max={numPages || 1}
              onChange={(e) => goToPage(parseInt(e.target.value, 10) || 1)}
            />
            <span> / {numPages || "..."}</span>
          </span>
          <button
            className="pdf-nav-btn"
            disabled={pageNumber >= (numPages || 1)}
            onClick={() => goToPage(pageNumber + 1)}
            title="Next page"
          >
            <ChevronRight size={16} />
          </button>
        </div>

        <div className="pdf-viewer-zoom">
          <button
            className="pdf-nav-btn"
            onClick={() => adjustScale(-0.25)}
            title="Zoom out"
          >
            <Minus size={14} />
          </button>
          <span className="pdf-zoom-label">{Math.round(scale * 100)}%</span>
          <button
            className="pdf-nav-btn"
            onClick={() => adjustScale(0.25)}
            title="Zoom in"
          >
            <Plus size={14} />
          </button>
        </div>

        {allPagesWithSources.size > 0 && (
          <div className="pdf-source-pages">
            {[...allPagesWithSources].sort((a, b) => a - b).map((p) => (
              <button
                key={p}
                className={`pdf-source-page-btn ${p === pageNumber ? "active" : ""}`}
                onClick={() => goToPage(p)}
                title={`Page ${p} has sources`}
              >
                p.{p}
              </button>
            ))}
          </div>
        )}

        <button className="pdf-close-btn" onClick={onClose} title="Close PDF viewer">
          <X size={16} />
        </button>
      </div>

      <div className="pdf-viewer-content" ref={containerRef}>
        <Document
          file={pdfUrl}
          onLoadSuccess={onDocumentLoadSuccess}
          loading={<div className="pdf-loading">Loading PDF...</div>}
          error={<div className="pdf-error">Failed to load PDF.</div>}
        >
          <div className="pdf-page-wrapper" ref={pageRef}>
            <Page
              pageNumber={pageNumber}
              scale={scale}
              onLoadSuccess={onPageLoadSuccess}
              renderTextLayer={true}
              renderAnnotationLayer={true}
            />

            {pageDimensions &&
              currentPageSources.map((src) => {
                const rect = transformBounds(src.bounds, pageDimensions, scale);
                if (!rect) return null;
                const isFocused = src.id === focusedSource;
                const colorIdx = (src.id - 1) % HIGHLIGHT_COLORS.length;
                return (
                  <div
                    key={`hl-${src.id}`}
                    className={`pdf-highlight ${isFocused ? "focused" : ""}`}
                    style={{
                      position: "absolute",
                      left: `${rect.left}px`,
                      top: `${rect.top}px`,
                      width: `${rect.width}px`,
                      height: `${rect.height}px`,
                      backgroundColor: isFocused
                        ? FOCUSED_COLORS[colorIdx]
                        : HIGHLIGHT_COLORS[colorIdx],
                      border: isFocused
                        ? `2px solid ${BORDER_COLORS[colorIdx]}`
                        : `1px solid ${BORDER_COLORS[colorIdx]}`,
                      cursor: "pointer",
                      pointerEvents: "auto",
                      zIndex: isFocused ? 11 : 10,
                    }}
                    onClick={() => onHighlightClick?.(src.id)}
                    title={`[${src.id}] ${src.source || ""} (${src.reference || ""})`}
                  >
                    <span className="pdf-highlight-label">[{src.id}]</span>
                  </div>
                );
              })}
          </div>
        </Document>
      </div>
    </div>
  );
}
