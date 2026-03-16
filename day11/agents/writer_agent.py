import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Map GEMINI_API_KEY to GOOGLE_API_KEY for LangChain if needed
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

async def writer_agent(state):

    question = state["question"]

    research = state.get("research", "")
    knowledge = state.get("knowledge", "")

    prompt = f"""
You are a factual and concise AI assistant. Your goal is to answer the user's question using ONLY the provided research and knowledge.

STRICT CONSTRAINTS:
1. USE ONLY PROVIDED CONTEXT: Do not use any outside knowledge. Use only what is in the "Research" and "Knowledge" sections.
2. Generate the final answers based completely on provided "knowledge" and "research".
3. Generate final answer after reading and understanding the provided "knowledge" and "research"
4. CITATION RULES: 
   - Every claim must be cited.
   - For "Knowledge (Local Documents)", use: [Source: filename, Page: X].
   - For "Research (Web Search)", use the URL provided in the search result: [Source: URL].

Question:
{question}

Research (Web Search):
{research}

Knowledge (Local Documents):
{knowledge}

Instruction: Write the answer based strictly on the above. If certain parts of the question cannot be answered, mention that those specific parts are missing from the data.

Final Answer:
"""

    response = await llm.ainvoke(prompt)

    state["final"] = response.content

    return state