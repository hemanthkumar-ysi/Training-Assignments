# chatbot.py

import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Create client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"  # Stable + fast

def get_response(user_input: str):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_input.strip()
    )

    reply = response.text
    usage = response.usage_metadata

    return reply, usage


def main():
    print("Gemini LLM Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        try:
            reply, usage = get_response(user_input)

            print("\nAssistant:", reply)

            if usage:
                print("\n--- Token Usage ---")
                print(f"Prompt Tokens: {usage.prompt_token_count}")
                print(f"Completion Tokens: {usage.candidates_token_count}")
                print(f"Total Tokens: {usage.total_token_count}")
                print("-------------------\n")

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()