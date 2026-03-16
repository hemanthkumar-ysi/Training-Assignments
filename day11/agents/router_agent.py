import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

async def router_agent(state):
    """
    Checks if the knowledge retrieved from the VectorDB is sufficient to answer the user's question.
    """
    question = state["question"]
    knowledge = state.get("knowledge", "")

    if not knowledge or "Knowledge base tool not available" in knowledge:
        state["action"] = "web_search"
        return state

    prompt = f"""
You are a verification agent. Your job is to determine if the provided "Retrieved Knowledge" is sufficient to fully and accurately answer the "User Question".

User Question:
{question}

Retrieved Knowledge:
{knowledge}

Decision Criteria:
1. If the retrieved knowledge contains the specific information needed to answer the question, return "sufficient".
2. If the retrieved knowledge is irrelevant, incomplete, or doesn't directly address the question, return "insufficient".

Return ONLY valid JSON:
{{
"sufficiency": "sufficient" or "insufficient",
"reason": "brief explanation"
}}
"""

    response = await llm.ainvoke(prompt)
    text = response.content.strip()

    try:
        # Simplified extraction in case of markdown blocks
        if "```json" in text:
            text = text.split("```json")[-1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[-1].split("```")[0].strip()
            
        decision = json.loads(text)
        if decision.get("sufficiency") == "insufficient":
            state["knowledge"]=""
            state["action"] = "web_search"
        else:
            state["action"] = "final_answer"

        state["action"] = "web_search" if decision.get("sufficiency") == "insufficient" else "final_answer"
    except Exception as e:
        print(f"Error parsing router decision: {e}")
        # Default to web search if unsure
        state["action"] = "web_search"

    return state
