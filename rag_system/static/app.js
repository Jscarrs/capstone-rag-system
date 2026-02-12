/**
 * RAG Chatbot Frontend
 * Handles chat interactions and document uploads with the Flask backend
 */

// Generate a unique session ID for this browser tab
const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

// DOM Elements - Chat
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const messagesContainer = document.getElementById('messages');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-btn');
const btnText = sendBtn.querySelector('.btn-text');
const btnLoading = sendBtn.querySelector('.btn-loading');

// DOM Elements - Upload
const uploadDropzone = document.getElementById('upload-dropzone');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const documentList = document.getElementById('document-list');
const resetDbBtn = document.getElementById('reset-db-btn');

/**
 * Add a message to the chat UI
 */
function addMessage(content, isUser, sources = null, isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    if (isError) {
        messageDiv.classList.add('error-message');
    }

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    messageDiv.appendChild(contentDiv);

    // Add sources if provided
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';

        const sourcesTitle = document.createElement('div');
        sourcesTitle.className = 'sources-title';
        sourcesTitle.textContent = 'References:';
        sourcesDiv.appendChild(sourcesTitle);

        sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';

            const sourceText = document.createElement('span');

            const sourceId = document.createElement('span');
            sourceId.className = 'source-id';
            sourceId.textContent = `[${source.id}]`;

            const sourceMeta = document.createElement('span');
            const sourceName = source.source || 'unknown';
            const reference = source.reference || (source.chunk != null ? `chunk ${source.chunk}` : '');
            sourceMeta.textContent = ` ${sourceName}${reference ? ` (${reference})` : ''}`;

            sourceText.appendChild(sourceId);
            sourceText.appendChild(sourceMeta);
            sourceItem.appendChild(sourceText);

            if (source.preview) {
                const preview = document.createElement('span');
                preview.className = 'source-preview';
                preview.textContent = source.preview.substring(0, 100) + '...';
                sourceItem.appendChild(preview);

                // Add tooltip with full content
                const tooltip = document.createElement('div');
                tooltip.className = 'source-tooltip';
                tooltip.textContent = source.preview;
                sourceItem.appendChild(tooltip);

                // Mark as hoverable
                sourceItem.classList.add('has-tooltip');
            }

            sourcesDiv.appendChild(sourceItem);
        });

        messageDiv.appendChild(sourcesDiv);
    }

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Add loading indicator
 */
function addLoadingIndicator() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message';
    loadingDiv.id = 'loading-indicator';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';

    loadingDiv.appendChild(contentDiv);
    messagesContainer.appendChild(loadingDiv);
    scrollToBottom();
}

/**
 * Remove loading indicator
 */
function removeLoadingIndicator() {
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.remove();
    }
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Set loading state for UI
 */
function setLoading(isLoading) {
    userInput.disabled = isLoading;
    sendBtn.disabled = isLoading;
    btnText.style.display = isLoading ? 'none' : 'inline';
    btnLoading.style.display = isLoading ? 'inline' : 'none';
}

/**
 * Send message to backend
 */
async function sendMessage(message) {
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }

        return await response.json();
    } catch (error) {
        console.error('Chat API error:', error);
        throw error;
    }
}

/**
 * Clear chat history
 */
async function clearChat() {
    try {
        await fetch('/api/clear', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId
            })
        });

        // Clear UI
        messagesContainer.innerHTML = '';

        // Add welcome message back
        addMessage(
            'Hello! I can answer questions based on the ingested documents. What would you like to know?',
            false
        );
    } catch (error) {
        console.error('Clear error:', error);
        addMessage('Failed to clear chat history.', false, null, true);
    }
}

/**
 * Handle form submission
 */
async function handleSubmit(e) {
    e.preventDefault();

    const message = userInput.value.trim();
    if (!message) return;

    // Add user message to UI
    addMessage(message, true);

    // Clear input
    userInput.value = '';

    // Set loading state
    setLoading(true);
    addLoadingIndicator();

    try {
        const result = await sendMessage(message);
        removeLoadingIndicator();
        addMessage(result.answer, false, result.sources);
    } catch (error) {
        removeLoadingIndicator();
        addMessage(`Error: ${error.message}`, false, null, true);
    } finally {
        setLoading(false);
        userInput.focus();
    }
}

// Event listeners - Chat
chatForm.addEventListener('submit', handleSubmit);
clearBtn.addEventListener('click', clearChat);

// --- Upload Functionality ---

/**
 * Upload a document to the server
 */
async function uploadDocument(file) {
    showUploadStatus('Uploading and processing...', 'uploading');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Upload failed');
        }

        showUploadStatus(result.message, 'success');
        console.log('Upload success:', result);

        // Refresh the document list
        loadDocumentList();
    } catch (error) {
        console.error('Upload error:', error);
        showUploadStatus(`Upload failed: ${error.message}`, 'error');
    }
}

/**
 * Show upload status message
 */
function showUploadStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = 'upload-status ' + type;
    uploadStatus.hidden = false;
}

/**
 * Load and display the list of ingested documents
 */
async function loadDocumentList() {
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        const docs = data.documents || [];

        if (docs.length === 0) {
            documentList.innerHTML = '';
            resetDbBtn.hidden = true;
            return;
        }

        let html = '<div class="document-list-title">Ingested documents:</div>';
        docs.forEach(doc => {
            html += `<span class="document-item">${doc.name}</span>`;
        });
        documentList.innerHTML = html;
        resetDbBtn.hidden = false;
    } catch (error) {
        console.error('Failed to load document list:', error);
    }
}

/**
 * Reset the database (clear ChromaDB and uploaded documents)
 */
async function resetDatabase() {
    if (!confirm('This will delete all ingested documents and the vector database. Continue?')) {
        return;
    }

    showUploadStatus('Clearing database...', 'uploading');

    try {
        const response = await fetch('/api/reset-db', { method: 'POST' });
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Reset failed');
        }

        showUploadStatus(result.message, 'success');
        console.log('Database reset:', result);
        loadDocumentList();
    } catch (error) {
        console.error('Reset error:', error);
        showUploadStatus(`Reset failed: ${error.message}`, 'error');
    }
}

// Event listeners - Upload
uploadDropzone.addEventListener('click', () => fileInput.click());
resetDbBtn.addEventListener('click', resetDatabase);

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        uploadDocument(e.target.files[0]);
        fileInput.value = ''; // Reset so same file can be re-uploaded
    }
});

// Drag and drop handlers
uploadDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadDropzone.classList.add('drag-over');
});

uploadDropzone.addEventListener('dragleave', () => {
    uploadDropzone.classList.remove('drag-over');
});

uploadDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        uploadDocument(e.dataTransfer.files[0]);
    }
});

// Load document list on page load
loadDocumentList();

// Focus input on load
userInput.focus();

// Log session ID for debugging
console.log('RAG Chatbot initialized with session:', sessionId);
