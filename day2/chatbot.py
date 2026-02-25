import os
import json
from dotenv import load_dotenv
import google.generativeai as genai


# ===============================
# LOAD API KEY
# ===============================

def load_api_key():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    return api_key


# ===============================
# INITIALIZE GEMINI
# ===============================

def init_model(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model


# ===============================
# READ QUESTIONS
# ===============================

def read_questions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
        return questions

    except FileNotFoundError:
        raise FileNotFoundError("Questions file not found")

    except Exception as e:
        raise Exception(f"Error reading file: {e}")


# ===============================
# ASK GEMINI
# ===============================

def ask_llm(model, question):
    try:
        response = model.generate_content(
            question,
            generation_config={
                "temperature": 0.2
            }
        )

        # Safe extraction
        answer = response.text.strip()

        return answer

    except Exception as e:
        return f"ERROR: {str(e)}"


# ===============================
# SAVE OUTPUT
# ===============================

def save_as_txt(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(f"Q: {item['question']}\n")
            f.write(f"A: {item['answer']}\n\n")


def save_as_json(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


# ===============================
# MAIN PIPELINE
# ===============================

def main():
    try:
        api_key = load_api_key()
        model = init_model(api_key)

        questions = read_questions("questions.txt")

        results = []

        for q in questions:
            print(f"Processing: {q}")
            ans = ask_llm(model, q)

            results.append({
                "question": q,
                "answer": ans
            })

        save_as_txt(results, "output/answers.txt")
        save_as_json(results, "output/answers.json")

        print("✅ Completed Successfully")

    except Exception as e:
        print(f"Fatal Error: {e}")


if __name__ == "__main__":
    main()