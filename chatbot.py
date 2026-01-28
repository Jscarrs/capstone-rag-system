import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

def get_llm():
    """
    Initialize the LLM based on available configuration.
    Priority: LM Studio (local) > OpenAI > Google Gemini
    """
    lmstudio_url = os.getenv("LMSTUDIO_BASE_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    # LM Studio (local, no API key needed)
    if lmstudio_url:
        from langchain_openai import ChatOpenAI
        print(f"[Using LM Studio at {lmstudio_url}]")
        return ChatOpenAI(
            base_url=lmstudio_url,
            api_key="lm-studio",  # LM Studio doesn't need a real key
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
            "No LLM configured. Set LMSTUDIO_BASE_URL, OPENAI_API_KEY, or GOOGLE_API_KEY in your .env file."
        )

# Initialize the chat model
llm = get_llm()

# Store conversation history
chat_history = []

def chat():
    print("Chatbot initialized! Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to history
        chat_history.append(HumanMessage(content=user_input))

        # Get response from LLM with full chat history
        response = llm.invoke(chat_history)

        # Add AI response to history
        chat_history.append(response)

        print(f"\nBot: {response.content}\n")

if __name__ == "__main__":
    chat()
