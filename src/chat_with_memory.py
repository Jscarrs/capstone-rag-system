from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

print("Starting chat...")

llm = ChatOllama(
    model="llama3",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

history = ChatMessageHistory()

chain = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history",
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "default"}}
    )

    print("Bot:", response.content)
