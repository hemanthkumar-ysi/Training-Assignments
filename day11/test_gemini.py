import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

if "GEMINI_API_KEY" in os.environ:
    print("GEMINI_API_KEY found")
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
else:
    print("GEMINI_API_KEY NOT found")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print("Model initialized successfully")
except Exception as e:
    print(f"Error initializing model: {e}")
