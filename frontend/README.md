# Frontend (React + Vite)

## Requirements
- Keep frontend and backend separated.
- Use `VITE_API_BASE_URL` for all backend API calls.
- Support chat, upload, document list, clear chat, and reset database flows.
- Keep styling in CSS files (no inline CSS).

## Overview
This frontend is a React + Vite client for the RAG backend API.  
It talks to the Flask server through `/api/*` endpoints and does not serve backend logic.

## Prerequisites
- Node.js 18+ (recommended)
- Backend API running from `../rag_system/rag_chatbot.py`

## Setup
```bash
cd frontend
npm install
cp .env.example .env
```

Edit `.env`:
```bash
VITE_API_BASE_URL=http://localhost:8080
```

## Run
```bash
cd frontend
npm run dev
```

Default dev URL: `http://localhost:5173`

## Backend Integration
Make sure backend CORS allows the frontend origin:
- In project root `.env`, set `FRONTEND_ORIGIN=http://localhost:5173`
- Start backend from `../rag_system`:
```bash
python3 rag_chatbot.py
```

## NPM Scripts
- `npm run dev` - start Vite dev server
- `npm run build` - production build
- `npm run preview` - preview production build

## Frontend Structure
```text
frontend/
├── .env.example
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── App.jsx
    ├── apiClient.js
    ├── main.jsx
    └── styles.css
```
