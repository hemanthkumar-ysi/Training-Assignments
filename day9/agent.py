import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import web_search, write_file


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

tools = [web_search, write_file]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
You are a research assistant.

Steps:
1. Search the web for the topic
2. Gather important information
3. Summarize the topic with key points
4. Save results using file writer tool
"""
)

topic = input("Enter topic: ")

response = agent.invoke({
    "messages": [
        {"role": "user", "content": f"Research this topic: {topic}"}
    ]
})

print(response)