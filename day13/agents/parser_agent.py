import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0  # Deterministic for extraction
)


def parse_resume(text):
    """
    Parses resume text and extracts structured information in JSON format.
    """
    prompt = f"""
    You are an expert AI Resume Analyzer. Your task is to extract structured information from the following resume text.
    
    EXTRACT THE FOLLOWING SECTIONS:
    1. **Contact Information**: Name, Email, Phone, Location
    2. **Professional Summary**: A brief paragraph of their profile.
    3. **Skills**: List of hard skills, soft skills or tools.
    4. **Experience**: List of work experience (Job Title, Company, Date, Core Bullet Points/Achievements).
    5. **Education**: Degrees, Universities, Graduation Years.
    6. **Projects**: List of notable projects (Project Title, Technologies Used, Key Highlights/Impact).

    Return the output strictly in the following JSON structure:
    {{
        "contact_info": {{
            "name": "",
            "email": "",
            "phone": "",
            "location": ""
        }},
        "summary": "",
        "skills": [],
        "experience": [
            {{
                "title": "",
                "company": "",
                "duration": "",
                "achievements": []
            }}
        ],
        "education": [
            {{
                "degree": "",
                "institution": "",
                "year": ""
            }}
        ],
        "projects": [
            {{
                "title": "",
                "highlights": []
            }}
        ]
    }}


    Resume Text:
    {text}
    """

    try:
        response = llm.invoke(prompt)
        content = response.content
        
        # Clean response string if it has markdown formatting
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as e:
        return {{"error": f"Failed to parse resume: {e}", "raw_content": response.content if 'response' in locals() else ""}}

if __name__ == "__main__":
    # Test text
    sample_text = "John Doe\nSoftware Engineer\nEmail: john@example.com\nSkills: Python, AWS\nExperience: SDE at Google (2022-Pres)\nEducation: BS CS, Stanford."
    print(parse_resume(sample_text))
