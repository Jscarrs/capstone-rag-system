# Capstone RAG System

## Overview

This project is a capstone implementation of a Retrieval-Augmented Generation (RAG) system.
The system is being developed incrementally, starting with a local language model and conversational memory.

## Current Status

**Sprint 1:** Local LLM + LangChain conversational memory

- Running a local LLM via Ollama
- Wrapped with LangChain
- Supports multi-turn conversations with memory

## Tech Stack

- Python 3.11.XX
- LangChain
- Ollama (local LLM)

## Next Steps

- Document loading and preprocessing
- Embeddings and vector storage
- Full RAG pipeline

## How to Run This Project (Sprint 1)

### 1. Install Python

Install **Python 3.11.x (64-bit)** from the official website:

https://www.python.org/downloads/windows/

> Python 3.11 is the most stable and widely supported version for LangChain and local LLM tooling.
> Make sure to **check “Add Python to PATH”** during installation.

Verify installation:

```powershell
py --version
```

### 2. Install Ollama

Ollama is used to run the language model locally.

Download and install Ollama from:
https://ollama.com/download

After installation, restart your terminal or VS Code.

Verify installation:

```powershell
ollama --version
```

### 3. Clone the repository

```powershell
git clone https://github.com/Jscarrs/capstone-rag-system.git
cd capstone-rag-system
```

### 4. Create and activate a virtual environment

```powershell
py -m venv venv
venv\Scripts\activate
```

### 5. Install dependencies

```powershell
pip install -r requirements.txt
```

### 6. Download the local language model (one-time setup)

```powershell
ollama run llama3
```

Once the model finishes downloading and the chat prompt appears, exit with Ctrl + C.

### 7. Run the application

```powershell
python src/chat_with_memory.py
```

You should see

```powershell
Starting chat...
```
