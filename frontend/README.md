# Frontend — React + Vite

React + Vite client for the Academic Research RAG backend API. Communicates with the Flask server exclusively through `/api/*` endpoints. No backend logic lives here.

## Overview

| Feature | Detail |
|---------|--------|
| Framework | React 18 + Vite 6 |
| Styling | Vanilla CSS (`styles.css`) — no inline styles, no Tailwind |
| PDF Rendering | `react-pdf` (PDF.js wrapper) |
| Markdown Rendering | `react-markdown` + `remark-gfm` |
| Icons | `lucide-react` |
| Backend URL | `VITE_API_BASE_URL` in `frontend/.env` |

## Prerequisites

- Node.js 18+
- Backend running at `../rag_system/rag_chatbot.py`

## Setup

```bash
cd frontend
npm install
cp .env.example .env
```

Edit `frontend/.env`:

```bash
VITE_API_BASE_URL=http://localhost:8080
```

## Run

```bash
cd frontend
npm run dev      # Dev server at http://localhost:5173
npm run build    # Production bundle
npm run preview  # Preview production build
```

## Backend Integration

Make sure the backend allows the frontend origin via CORS:
- In project root `.env`: set `FRONTEND_ORIGIN=http://localhost:5173`
- Start backend from `../rag_system`:
  ```bash
  python3 rag_chatbot.py
  ```

## Key Flows

### Chat

`App.jsx` manages session state, message history (persisted in `sessionStorage`), and API calls. Sending a message:
1. POST `/api/chat` with `{ message, session_id }`
2. Response includes `answer` (Markdown text) + `sources` array
3. Citations like `[1]`, `[2]` in the answer become clickable spans — clicking opens the PDF viewer at that page
4. Source references expand in a collapsible `<details>` panel below each message
5. Assistant messages have a **Copy** button (copies plain text to clipboard)
6. Failed messages show a **Retry** button that re-sends the same question

### Document Upload

Drag-and-drop or file picker in the sidebar sends a file to POST `/api/upload`. On success, the document list is refreshed via GET `/api/documents`. Supports `.txt` and `.pdf`.

### PDF Reference Highlighting

When a user clicks a citation or a PDF document in the sidebar:
1. `PdfViewer.jsx` loads via `react-pdf` (lazy-imported with `React.lazy()` for performance)
2. Source bounding boxes returned from the API are overlaid as semi-transparent colored divs
3. Clicking a highlight or citation focuses that source (prominent highlight)
4. Source page buttons in the toolbar allow quick navigation between cited pages

Coordinate transform in `PdfViewer.jsx` (Adobe bottom-left → CSS top-left):
```
css_x = bounds_x_min × scale
css_y = (page_height − bounds_y_max) × scale
```

### Figure Lightbox

Clicking any inline figure image (shown below a message) or a source thumbnail opens a full-screen lightbox overlay. Clicking anywhere outside the image or the close button dismisses it.

### Actions

- **Clear Chat**: POST `/api/clear` + resets `sessionStorage`
- **Clear Database**: POST `/api/reset-db` — guarded by a **confirmation modal** before executing; also resets chat on success

### Toast Notifications

All async operations (upload, clear, reset) emit toast notifications (`success`, `error`, `info`) that auto-dismiss after 4 seconds. Toasts stack and can be manually dismissed.

### Splash Screen

On first visit, a welcome splash screen is shown explaining how to use the chatbot. Users can check "Do not show this again" to suppress it via `localStorage`. The splash can reappear after storage is cleared.


## Source Structure

```
frontend/
├── .env.example
├── index.html
├── package.json
├── vite.config.js          # PDF.js cMaps copied to dist for non-latin font support
└── src/
    ├── App.jsx             # Root component: chat, sidebar, upload, modals, toasts
    ├── PdfViewer.jsx       # Embedded PDF viewer with bounding-box highlight overlays
    ├── apiClient.js        # Centralized fetch wrappers (apiGet, apiPost, apiUpload)
    ├── main.jsx            # React entry point
    └── styles.css          # All styles (dark/light theme via data-theme attribute)
```

## Theme

Dark/light mode is toggled from the header. The current preference is persisted in `localStorage` and applied via `data-theme="dark"|"light"` on `<html>`.

## Session Persistence

Chat messages and the session ID are stored in `sessionStorage` (cleared on browser/tab close). A storage version key (`rag_storage_version`) is checked on startup to flush stale data after upgrades.

## PDF.js cMaps

`vite.config.js` uses `vite-plugin-static-copy` to copy `pdfjs-dist/cmaps` into the build output. This is required for rendering PDFs that contain non-latin characters (e.g., Chinese).
