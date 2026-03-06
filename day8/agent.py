import os
import json
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------
# TOOLS
# -------------------------

# Financial calculator
def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)


# Date / Time tool
def get_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Currency conversion tool (mock rates)
def convert_usd_to_inr(amount):
    rate = 83  # example conversion rate
    return str(amount * rate)


# Weather tool (mock)
def get_weather(city):
    weather_data = {
        "hyderabad": "32°C, Sunny",
        "delhi": "28°C, Cloudy",
        "mumbai": "30°C, Humid"
    }

    return weather_data.get(city.lower(), "Weather data not available")


# Text summarization tool
def summarize_text(text):

    prompt = f"""
Summarize the following text in 2 sentences:

{text}
"""

    response = model.generate_content(prompt)
    return response.text


# -------------------------
# AGENT LOOP
# -------------------------

def run_agent(query):

    scratchpad = ""

    while True:

        prompt = f"""
You are an AI agent that can solve user queries by reasoning and using tools when necessary.

Available tools:

1. calculator
Input: mathematical expression
Example: 2500*12

2. datetime
Input: none
Returns current date and time

3. summarize
Input: long text
Returns a short summary of the text

4. currency_converter
Input: amount in USD
Converts USD to INR
Example: 50

5. weather
Input: city name
Returns current weather information
Example: Hyderabad

Rules you MUST follow:

1. Decide whether a tool is needed.
2. If a tool is needed, respond ONLY in this format:

Action:
{{
 "tool": "<tool_name>",
 "input": "<input>"
}}

3. After receiving the tool result (Observation), you MUST provide the final answer.

4. Do NOT call the same tool repeatedly.

5. If the answer can be given directly, respond with:

Final Answer: <your answer>

6. You may use a tool at most ONE time for a query.

7. After a tool result is given, convert the result into a natural language final answer.


Previous steps:
{scratchpad}

User Question:
{query}
"""

        response = model.generate_content(prompt)
        output = response.text.strip()

        print("\nLLM Output:")
        print(output)

        # Final answer
        if "Final Answer:" in output:
            return output

        # Tool usage
        if "Action:" in output:

            action_json = output.split("Action:")[1].strip()
            action = json.loads(action_json)

            tool = action["tool"]
            tool_input = action.get("input", "")

            print("\nUsing Tool:", tool)
            print("Tool Input:", tool_input)

            # Tool execution
            if tool == "calculator":
                observation = calculator(tool_input)

            elif tool == "datetime":
                observation = get_datetime()

            elif tool == "currency_converter":
                observation = convert_usd_to_inr(float(tool_input))

            elif tool == "weather":
                observation = get_weather(tool_input)

            elif tool == "summarize":
                observation = summarize_text(tool_input)

            else:
                observation = "Unknown tool"

            print("Tool Output:", observation)

            scratchpad += f"""
Action: {action_json}
Observation: {observation}
"""


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":

    while True:

        user_input = input("\nAsk something: ")

        result = run_agent(user_input)

        print("\nFinal Result:")
        print(result)