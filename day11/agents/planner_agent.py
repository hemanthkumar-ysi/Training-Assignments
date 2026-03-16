from dotenv import load_dotenv
load_dotenv()
import json
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

async def planner_agent(state):

    question = state["question"]

    prompt = f"""
You are a Search Optimization Agent. Your job is to take a user's question and reframe it into a concise, high-quality search query for a Vector Database/Knowledge Base only if the user query is not good enough.

GUIDELINES:
1. Strip away all conversational filler (e.g., "please", "can you tell me", "I was wondering if").
2. Focus on core semantic keywords and technical terms.Do not change the context of question by any chance.
3. If the question is already a perfect search query, return it as is.
4. If the question is complex, break it down into the most important searchable concept.

EXAMPLES:
- User: "Hey, can you please find me some information about the latest safety protocols for lithium batteries in 2024?"
  Query: "lithium battery safety protocols 2024"
- User: "What is the difference between a transformer and a recruiter in deep learning? I am confused."
  Query: "transformer vs recruiter deep learning architecture"
- User: "Explain the project structure of this repository."
  Query: "project structure overview"
- User: "when did India get Independence"
  Query: "date and Year India got Independence"

Question:
{question}

Return ONLY valid JSON:
{{
"question": "user/reframed question that needs to be returned"
}}
"""

    response = await llm.ainvoke(prompt)

    text = response.content.strip()

    try:
        decision = json.loads(text)
    except:
        decision = {
            "topic": question
        }
    
    state["action"] = "knowledge_base"
    state["topic"] = decision.get("question", question)
    print("searching for topic:",state["topic"])

    return state