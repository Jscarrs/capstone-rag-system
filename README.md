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
- Python 3.9+
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
