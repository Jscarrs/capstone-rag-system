"""
Shared Utilities

Centralized provider configuration and text processing used by
rag_chatbot.py, ingest.py, and ingest_single_file.py.

Environment Variables:
- LMSTUDIO_BASE_URL: Local LM Studio server URL
- OPENAI_API_KEY: OpenAI API key
- GOOGLE_API_KEY: Google Gemini API key
- USE_LOCAL_EMBEDDINGS: Use local HuggingFace embeddings
"""

import os
from dotenv import load_dotenv

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = os.path.join(_parent_dir, ".env")
load_dotenv(_env_path)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "assets", "figures")


def get_embeddings():
    """
    Initialize embeddings based on available configuration.
    Priority: HuggingFace (local) > OpenAI > Google Gemini
    """
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    lmstudio_url = os.getenv("LMSTUDIO_BASE_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if use_local or lmstudio_url:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("[Using Local HuggingFace Embeddings (all-MiniLM-L6-v2)]")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    elif openai_key and openai_key != "your_openai_api_key_here":
        from langchain_openai import OpenAIEmbeddings
        print("[Using OpenAI Embeddings]")
        return OpenAIEmbeddings(openai_api_key=openai_key)
    elif google_key and google_key != "your_google_api_key_here":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        print("[Using Google Gemini Embeddings]")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_key,
        )
    else:
        raise ValueError(
            "No embeddings configured. Set USE_LOCAL_EMBEDDINGS=true, "
            "OPENAI_API_KEY, or GOOGLE_API_KEY in your .env file."
        )


def get_llm():
    """
    Initialize the LLM based on available configuration.
    Priority: LM Studio (local) > OpenAI > Google Gemini
    """
    lmstudio_url = os.getenv("LMSTUDIO_BASE_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if lmstudio_url:
        from langchain_openai import ChatOpenAI
        print(f"[Using LM Studio at {lmstudio_url}]")
        return ChatOpenAI(
            base_url=lmstudio_url,
            api_key="lm-studio",
            temperature=0.7
        )
    elif openai_key and openai_key != "your_openai_api_key_here":
        from langchain_openai import ChatOpenAI
        print("[Using OpenAI GPT-3.5-turbo]")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=openai_key
        )
    elif google_key and google_key != "your_google_api_key_here":
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("[Using Google Gemini 2.5 Flash]")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=google_key
        )
    else:
        raise ValueError(
            "No LLM configured. Set LMSTUDIO_BASE_URL, OPENAI_API_KEY, "
            "or GOOGLE_API_KEY in your .env file."
        )


def split_text(text, chunk_size=1000, chunk_overlap=300):
    """
    Split text into overlapping chunks with position tracking.

    Overlap of 300 ensures figure references (like 'see Figure 2')
    are captured alongside descriptive text.
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunks.append({"text": text[start:end], "start_pos": start})
        start += chunk_size - chunk_overlap

    return chunks
