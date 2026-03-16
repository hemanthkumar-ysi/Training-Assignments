import os
import sys

print("Python version:", sys.version)
print("Current Working Directory:", os.getcwd())

print("Loading dotenv...")
from dotenv import load_dotenv
load_dotenv()
print("Dotenv loaded")

if "GEMINI_API_KEY" in os.environ:
    print("GEMINI_API_KEY found")
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
else:
    print("GEMINI_API_KEY NOT found")

print("Importing langchain_google_genai...")
from langchain_google_genai import ChatGoogleGenerativeAI
print("Imported ChatGoogleGenerativeAI")

try:
    print("Initializing model...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print("Model initialized successfully!")
    
    print("Invoking model...")
    response = llm.invoke("Hello, what is your name?")
    print("Response received:")
    print(response.content)
except Exception as e:
    print(f"Error during execution: {e}")

print("Test script finished.")
