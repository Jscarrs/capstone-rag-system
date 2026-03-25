
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { 
  Send, UploadCloud, Trash2, RefreshCw, X, AlertCircle, 
  CheckCircle, Info, Bot, User, FileText, Loader2, File,
  Image, ZoomIn
} from "lucide-react";
import { API_BASE_URL, apiGet, apiPost, apiUpload } from "./apiClient";

const ALLOWED_EXTENSIONS = [".txt", ".pdf"];
const WELCOME_TEXT = "Hello! Please insert a document and ask questions about its content.";

function createMessage(role, content, sources = []) {
  return {
    id: `${Date.now()}_${Math.random().toString(36).slice(2, 10)}`,
    role,
    content,
    sources
  };
}

function createSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
}

function formatSourceLabel(source) {
  const name = source.source || "unknown";
  const reference = source.reference || (source.chunk != null ? `chunk ${source.chunk}` : "");
  return reference ? `${name} (${reference})` : name;
}

function getFileExtension(filename) {
  const dotIndex = filename.lastIndexOf(".");
  return dotIndex >= 0 ? filename.slice(dotIndex).toLowerCase() : "";
}

function buildImageUrl(imageUrl) {
  if (!imageUrl) return null;
  return `${API_BASE_URL}${imageUrl}`;
}

function buildFigureMap(sources) {
  const map = {};
  if (!sources) return map;
  for (const s of sources) {
    if (s.image_url) {
      map[s.id] = buildImageUrl(s.image_url);
    }
  }
  return map;
}

const CITATION_RE = /\[(\d+)\]/g;

function MessageContent({ text, sources, onImageClick }) {
  const figureMap = useMemo(() => buildFigureMap(sources), [sources]);

  if (Object.keys(figureMap).length === 0) {
    return <p className="message-text">{text}</p>;
  }

  const parts = [];
  let lastIndex = 0;
  const matches = [...text.matchAll(CITATION_RE)];

  for (const match of matches) {
    if (match.index > lastIndex) {
      parts.push({ type: "text", value: text.slice(lastIndex, match.index) });
    }
    const citationId = parseInt(match[1], 10);
    parts.push({ type: "citation", value: match[0], id: citationId });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push({ type: "text", value: text.slice(lastIndex) });
  }

  return (
    <div className="message-text">
      {parts.map((part, i) => {
        if (part.type === "text") {
          return <span key={i}>{part.value}</span>;
        }
        const imageUrl = figureMap[part.id];
        return (
          <span key={i}>
            <span className="citation-ref">{part.value}</span>
            {imageUrl && (
              <span className="inline-figure" onClick={() => onImageClick(imageUrl)}>
                <img src={imageUrl} alt={`Figure [${part.id}]`} loading="lazy" />
                <span className="inline-figure-zoom"><ZoomIn size={14} /></span>
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}

export default function App() {
  const sessionId = useMemo(createSessionId, []);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const [messages, setMessages] = useState([createMessage("assistant", WELCOME_TEXT)]);
  const [inputValue, setInputValue] = useState("");
  const [isSending, setIsSending] = useState(false);

  const [documents, setDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);

  // Lightbox for figure images
  const [lightboxUrl, setLightboxUrl] = useState(null);

  // Toast System
  const [toasts, setToasts] = useState([]);
  const addToast = useCallback((type, message) => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { id, type, message }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const removeToast = (id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  // Modal System
  const [confirmModal, setConfirmModal] = useState({ 
    isOpen: false, 
    onConfirm: null, 
    title: "", 
    message: "" 
  });

  // Splash Screen State
  const [showSplash, setShowSplash] = useState(() => {
    return localStorage.getItem("hideSplash") !== "true";
  });
  const [dontShowAgain, setDontShowAgain] = useState(false);

  function handleCloseSplash() {
    if (dontShowAgain) {
      localStorage.setItem("hideSplash", "true");
    }
    setShowSplash(false);
  }

  useEffect(() => {
    void loadDocuments();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [inputValue]);

  async function loadDocuments() {
    try {
      const response = await apiGet("/api/documents");
      const docs = response.documents || [];
      setDocuments(docs);
      console.log("[documents] loaded", docs.length);
    } catch (error) {
      console.error("[documents] load failed:", error);
      addToast("error", `Failed to load documents: ${error.message}`);
    }
  }

  function appendMessage(role, content, sources = []) {
    setMessages((previous) => [...previous, createMessage(role, content, sources)]);
  }

  async function handleSend(event) {
    if (event) event.preventDefault();
    const question = inputValue.trim();
    if (!question || isSending) {
      return;
    }

    appendMessage("user", question);
    setInputValue("");
    setIsSending(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    console.log("[chat] sending question");

    try {
      const result = await apiPost("/api/chat", {
        message: question,
        session_id: sessionId
      });
      appendMessage("assistant", result.answer, result.sources || []);
      console.log("[chat] response received");
    } catch (error) {
      console.error("[chat] failed:", error);
      appendMessage("error", `Error: ${error.message}`);
    } finally {
      setIsSending(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  }

  async function handleClearChat() {
    try {
      await apiPost("/api/clear", { session_id: sessionId });
      setMessages([createMessage("assistant", WELCOME_TEXT)]);
      console.log("[chat] cleared");
      addToast("success", "Chat cleared successfully.");
    } catch (error) {
      console.error("[chat] clear failed:", error);
      addToast("error", `Failed to clear chat: ${error.message}`);
    }
  }

  async function handleUpload(file) {
    if (!file || isUploading) {
      return;
    }

    const extension = getFileExtension(file.name);
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      const allowed = ALLOWED_EXTENSIONS.join(", ");
      addToast("error", `Unsupported file type ${extension || "(none)"}. Allowed: ${allowed}`);
      return;
    }

    setIsUploading(true);
    addToast("info", `Uploading ${file.name}...`);
    console.log("[upload] start", file.name);

    try {
      const result = await apiUpload("/api/upload", file);
      addToast("success", result.message || `Uploaded ${file.name}`);
      console.log("[upload] success", file.name);
      await loadDocuments();
    } catch (error) {
      console.error("[upload] failed:", error);
      addToast("error", `Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  }

  function requestResetDatabase() {
    setConfirmModal({
      isOpen: true,
      title: "Clear Database?",
      message: "This will delete all ingested documents and clear the vector database. This action cannot be undone.",
      onConfirm: async () => {
        setConfirmModal((prev) => ({ ...prev, isOpen: false }));
        await performResetDatabase();
      }
    });
  }

  async function performResetDatabase() {
    if (isResetting) return;

    setIsResetting(true);
    addToast("info", "Clearing database...");
    console.log("[reset] requested");

    try {
      const result = await apiPost("/api/reset-db");
      addToast("success", result.message || "Database cleared.");
      await loadDocuments();
      console.log("[reset] success");
    } catch (error) {
      console.error("[reset] failed:", error);
      addToast("error", `Reset failed: ${error.message}`);
    } finally {
      setIsResetting(false);
    }
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function onFileChange(event) {
    const file = event.target.files?.[0];
    if (file) {
      void handleUpload(file);
      event.target.value = "";
    }
  }

  function onDrop(event) {
    event.preventDefault();
    setIsDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      void handleUpload(file);
    }
  }

  function onDragOver(event) {
    event.preventDefault();
    setIsDragActive(true);
  }

  function onDragLeave(event) {
    event.preventDefault();
    setIsDragActive(false);
  }

  const canSend = inputValue.trim().length > 0 && !isSending;

  return (
    <>
      <div className="app-shell">
        <header className="app-header">
          <div>
            <p className="eyebrow">Team Station 4</p>
            <h1>Academic Research RAG</h1>
            <p className="subtitle">
              Intelligent retrieval for academic papers and research data.
            </p>
          </div>
          <div className="meta-block">
            <span className="meta-label">Backend</span>
            <code className="meta-value">{API_BASE_URL}</code>
          </div>
        </header>

        <main className="layout">
          <aside className="side-panel">
            <section className="panel-card">
              <h2>Document Ingestion</h2>
              <p className="panel-help">Drop a .txt or .pdf file, or click to browse.</p>
              <button
                type="button"
                className={`dropzone ${isDragActive ? "drag-active" : ""}`}
                onClick={openFilePicker}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                disabled={isUploading}
              >
                {isUploading ? (
                  <Loader2 className="dropzone-icon animate-spin" size={32} />
                ) : (
                  <UploadCloud className="dropzone-icon" size={32} />
                )}
                <span className="dropzone-title">
                  {isUploading ? "Uploading..." : "Upload Document"}
                </span>
                <span className="dropzone-hint">Supports .txt and .pdf</span>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.pdf"
                onChange={onFileChange}
                className="hidden-input"
              />
            </section>

            <section className="panel-card">
              <div className="panel-title-row">
                <h2>Ingested Documents</h2>
                <span className="count-pill">{documents.length}</span>
              </div>
              {documents.length === 0 ? (
                <div className="empty-note">
                  <FileText size={32} opacity={0.3} />
                  <p>No files ingested yet.</p>
                </div>
              ) : (
                <ul className="doc-list">
                  {documents.map((doc) => (
                    <li key={doc.name} className="doc-item">
                      <div className="doc-name-wrap">
                        <File size={14} className="text-ink-500" />
                        <span className="doc-name" title={doc.name}>{doc.name}</span>
                      </div>
                      <span className="doc-meta">{doc.type.toUpperCase()}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className="panel-card">
              <h2>Actions</h2>
              <div className="actions">
                <button type="button" className="secondary-btn" onClick={handleClearChat}>
                  <RefreshCw size={16} />
                  Clear Chat
                </button>
                <button
                  type="button"
                  className="danger-btn"
                  onClick={requestResetDatabase}
                  disabled={isResetting}
                >
                  {isResetting ? <Loader2 size={16} className="animate-spin" /> : <Trash2 size={16} />}
                  {isResetting ? "Clearing..." : "Clear Database"}
                </button>
              </div>
            </section>
          </aside>

          <section className="chat-panel">
            <div className="messages">
              {messages.map((message) => (
                <div key={message.id} className={`message-row ${message.role}`}>
                  <div className={`message-avatar ${message.role === 'user' ? 'user-avatar' : ''}`}>
                    {message.role === 'user' ? (
                      <User size={20} />
                    ) : (
                      <img src="/tmu-bot-avatar.png" alt="Bot Avatar" onError={(e) => {
                        e.target.onerror = null; 
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'block';
                      }} />
                    )}
                    {message.role !== 'user' && <Bot size={20} style={{ display: 'none' }} />}
                  </div>
                  
                  <article className="message-card">
                    <MessageContent
                      text={message.content}
                      sources={message.sources}
                      onImageClick={setLightboxUrl}
                    />
                    {message.sources && message.sources.length > 0 ? (
                      <details className="source-box">
                        <summary>References ({message.sources.length})</summary>
                        <ul className="source-list">
                          {message.sources.map((source) => (
                            <li key={`${message.id}_${source.id}`} className="source-item">
                              <p className="source-title">
                                {source.image_url ? (
                                  <Image size={14} className="source-icon" />
                                ) : (
                                  <FileText size={14} className="source-icon" />
                                )}
                                <span className="source-id">[{source.id}]</span>{" "}
                                {formatSourceLabel(source)}
                              </p>
                              {source.image_url ? (
                                <div
                                  className="source-thumbnail"
                                  onClick={() => setLightboxUrl(buildImageUrl(source.image_url))}
                                >
                                  <img
                                    src={buildImageUrl(source.image_url)}
                                    alt={formatSourceLabel(source)}
                                    loading="lazy"
                                  />
                                  <span className="source-thumbnail-zoom"><ZoomIn size={14} /></span>
                                </div>
                              ) : source.preview ? (
                                <p className="source-preview">{source.preview}</p>
                              ) : null}
                            </li>
                          ))}
                        </ul>
                      </details>
                    ) : null}
                  </article>
                </div>
              ))}

              {isSending ? (
                <div className="message-row assistant">
                  <div className="message-avatar">
                    <img src="/tmu-bot-avatar.png" alt="Bot Avatar" onError={(e) => {
                      e.target.onerror = null; 
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'block';
                    }} />
                    <Bot size={20} style={{ display: 'none' }} />
                  </div>
                  <article className="message-card">
                    <p className="message-text loading-line">
                      <span />
                      <span />
                      <span />
                    </p>
                  </article>
                </div>
              ) : null}

              <div ref={messagesEndRef} />
            </div>

            <form className="chat-form" onSubmit={handleSend}>
              <label htmlFor="chat-input" className="sr-only">
                Message
              </label>
              <textarea
                id="chat-input"
                ref={textareaRef}
                value={inputValue}
                onChange={(event) => setInputValue(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="What would you like to know?"
                autoComplete="off"
                disabled={isSending}
                rows={1}
              />
              <button type="submit" className="primary-btn" disabled={!canSend}>
                <Send size={16} />
                {isSending ? "Sending" : "Send"}
              </button>
            </form>
          </section>
        </main>

        <footer className="app-footer">
          <p>&copy; {new Date().getFullYear()} Team Station 4. All rights reserved.</p>
        </footer>
      </div>

      {/* Toast Notifications */}
      <div className="toast-container">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            {toast.type === "success" && <CheckCircle size={20} className="toast-icon" />}
            {toast.type === "error" && <AlertCircle size={20} className="toast-icon" />}
            {toast.type === "info" && <Info size={20} className="toast-icon" />}
            <span className="toast-content">{toast.message}</span>
            <button className="toast-close" onClick={() => removeToast(toast.id)}>
              <X size={16} />
            </button>
          </div>
        ))}
      </div>

      {/* Splash Screen */}
      {showSplash && (
        <div className="modal-overlay" style={{ zIndex: 200 }}>
          <div className="modal-content splash-content">
            <div className="splash-header">
              <Bot size={48} className="splash-icon" />
              <h2>Welcome to Academic Research RAG</h2>
            </div>
            <div className="splash-body">
              <p>
                This intelligent assistant helps you quickly retrieve and synthesize information from your academic papers and research data.
              </p>
              <ul>
                <li><strong>Upload:</strong> Add your PDF or TXT documents to the knowledge base.</li>
                <li><strong>Query:</strong> Ask complex questions in natural language.</li>
                <li><strong>Discover:</strong> Get synthesized answers with direct citations to your uploaded sources.</li>
              </ul>
            </div>
            <div className="splash-footer">
              <label className="checkbox-label">
                <input 
                  type="checkbox" 
                  checked={dontShowAgain}
                  onChange={(e) => setDontShowAgain(e.target.checked)}
                />
                Do not show this again
              </label>
              <button type="button" className="primary-btn" onClick={handleCloseSplash}>
                Get Started
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Figure Lightbox */}
      {lightboxUrl && (
        <div className="lightbox-overlay" onClick={() => setLightboxUrl(null)}>
          <button className="lightbox-close" onClick={() => setLightboxUrl(null)}>
            <X size={24} />
          </button>
          <img
            className="lightbox-image"
            src={lightboxUrl}
            alt="Figure"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}

      {/* Confirmation Modal */}
      {confirmModal.isOpen && (
        <div className="modal-overlay" onClick={() => setConfirmModal(prev => ({ ...prev, isOpen: false }))}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <AlertCircle size={24} className="modal-icon" />
              <h3>{confirmModal.title}</h3>
            </div>
            <div className="modal-body">
              <p>{confirmModal.message}</p>
            </div>
            <div className="modal-actions">
              <button 
                type="button" 
                className="secondary-btn" 
                onClick={() => setConfirmModal(prev => ({ ...prev, isOpen: false }))}
              >
                Cancel
              </button>
              <button 
                type="button" 
                className="danger-btn" 
                onClick={confirmModal.onConfirm}
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
