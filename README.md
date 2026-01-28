# LangChain Chatbot with Memory

A simple chatbot built with LangChain that maintains conversation memory. Supports both OpenAI and Google Gemini.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file and add your API key:
```bash
cp .env.example .env
```

Then edit `.env` and add your actual API key (choose one):
```
OPENAI_API_KEY=sk-your-actual-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

## Usage

Run the chatbot:
```bash
python chatbot.py
```

The chatbot will remember the conversation context as you chat. Type 'quit' or 'exit' to end the conversation.

## Features

- Supports OpenAI GPT-3.5-turbo or Google Gemini 2.0 Flash
- Automatically detects which API key is configured
- Maintains conversation memory throughout the session
- Simple command-line interface
