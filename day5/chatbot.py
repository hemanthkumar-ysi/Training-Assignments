import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# -------- Load API Key --------
load_dotenv()

# -------- LLM --------
# Using the stable 1.5-flash model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# -------- Manual Memory (State) --------
# We use a simple list to store message objects
chat_history = []

# -------- Prompt --------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and context-aware assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# -------- Chain --------
chain = prompt | llm | StrOutputParser()

# -------- Chat Loop --------
print("Chatbot started (type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    if user_input.lower() in ["history"]:
        print(chat_history)
        break

    # Invoke the chain by passing the current history list
    response = chain.invoke({
        "input": user_input,
        "history": chat_history
    })

    print(f"\nBot: {response}\n")

    # Update chat history manually
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))