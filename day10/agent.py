import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
    
)


# Search Tool
search_tool = DuckDuckGoSearchRun()


# State
class AgentState(TypedDict):
    question: str
    thought: str
    search_needed: bool
    search_result: str
    answer: str


# Node 1 — Think
def think_node(state):
    print("\n--- THINK NODE ---")
    print("Current State:", state)

    question = state["question"]

    response = llm.invoke(f"""
    You are an AI researcher.

    Question: {question}

    Decide if web search is required.

    Return format:
    Thought: <reasoning>
    SearchNeeded: yes or no
    """)

    text = response.content
    search_needed = "searchneeded: yes" in text.lower()

    return {
        "thought": text,
        "search_needed": search_needed
    }


# Node 2 — Search
def search_node(state):
    print("\n--- SEARCH NODE ---")
    print("Current State:", state)

    query = state["question"] + " " + state.get("thought", "")

    result = search_tool.run(query)

    return {
        "search_result": result
    }


# Node 3 — Final Answer
def answer_node(state):
    print("\n--- ANSWER NODE ---")
    print("Current State:", state)


    question = state["question"]
    thought = state.get("thought", "")
    search_result = state.get("search_result", "")

    response = llm.invoke(f"""
    Question: {question}

    Reasoning: {thought}

    Search Results: {search_result}

    Provide a clear final answer.
    """)
    return {
        "answer": response.content
    }


# Decision Function
def decide_next(state):

    if state["search_needed"]:
        return "search"

    return "answer"


# Build Graph
graph = StateGraph(AgentState)

graph.add_node("think", think_node)
graph.add_node("search", search_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("think")

graph.add_conditional_edges(
    "think",
    decide_next,
    {
        "search": "search",
        "answer": "answer"
    }
)

graph.add_edge("search", "answer")
graph.add_edge("answer", END)

app = graph.compile()

q=str(input("ask a question:"))
# Run Agent
result = app.invoke({
    "question": q
})

print(result["answer"])