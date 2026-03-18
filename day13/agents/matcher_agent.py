import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0  # Enforce strict formatting layout
)



def match_resume_to_job(structured_resume, job_description):
    """
    Matches the structured resume data against a target Job Description.
    """
    prompt = f"""
    You are an expert AI Recruiter. Match the following high-value candidate resume dataset against the target Job Description (JD).
    
    Candidate Data:
    {structured_resume}

    Target Job Description:
    {job_description}

    Instructions:
    1. **Calculate a Match Score (0-100)**: Compute the TOTAL score by summing the following weighted sub-scores:
        - **Skills (40 points max)**: Calculate % overlap with required technical skills mentioned in the JD.
        - **Experience (30 points max)**: Rate based on years of relevant domain depth and leadership vs JD.
        - **Projects (20 points max)**: Rate relative alignment of past built tools/apps to the target company's niche.
        - **Education & Certs (10 points max)**: Rate alignment of diplomas/certificates to guidelines.
    
    2. **Missing Keywords**: Name important technical or soft skills mentioned in the JD that are not explicitly found in the resume.
    3. **Strengths for this role**: Why they fit this specific company or description properly.
    4. **Recommendations for Tailoring**: 2 quick bullets on what they should add or change in the resume to secure the interview for *this* position.

    Output Rules:
    - **CRITICAL**: Return ONLY the JSON. No conversational text.
    - **JSON Structure must be Valid RFC 8259**: All keys and strings must use double quotes `""`. No single quotes.
    - **No Trailing Commas**: Ensure keys/lists do not end with a trailing comma.

    Return the response strictly in the following JSON format structure:
    {{
        "match_score": 0,
        "score_breakdown": {{
            "skills": 0,
            "experience": 0,
            "projects": 0,
            "education": 0
        }},
        "missing_keywords": [],
        "strengths": "",
        "tailoring_advice": []
    }}


    """

    try:
        response = llm.invoke(prompt)
        content = response.content

        # Clean response if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)
    except Exception as e:
        # Fix: Using single braces for normal Python dicts to solve unhashable dict type error
        raw_text = response.content if 'response' in locals() else ""
        return {"error": f"Failed to match resume: {e}", "raw_content": raw_text}


if __name__ == "__main__":
    resume = {{"skills": ["Python", "AWS"], "experience": []}}
    jd = "Required: Python Developer with AWS and Docker experience."
    print(match_resume_to_job(resume, jd))
