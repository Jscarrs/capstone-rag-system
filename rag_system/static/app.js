/**
 * RAG Chatbot Frontend
 * Handles chat interactions with the Flask backend
 */

// Generate a unique session ID for this browser tab
const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

// DOM Elements
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const messagesContainer = document.getElementById('messages');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-btn');
const btnText = sendBtn.querySelector('.btn-text');
const btnLoading = sendBtn.querySelector('.btn-loading');

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
            sourceText.innerHTML = `<span class="source-id">[${source.id}]</span> ${source.source} (chunk ${source.chunk})`;
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

// Event listeners
chatForm.addEventListener('submit', handleSubmit);
clearBtn.addEventListener('click', clearChat);

// Focus input on load
userInput.focus();

// Log session ID for debugging
console.log('RAG Chatbot initialized with session:', sessionId);
