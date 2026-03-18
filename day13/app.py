import streamlit as st
import os
import json

# Silence duplicate API key warnings from LangChain
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" in os.environ:
    os.environ.pop("GOOGLE_API_KEY", None)

from agents.matcher_agent import match_resume_to_job
from tools.pdf_extractor import extract_text_from_pdf  # Top level imports

from agents.parser_agent import parse_resume

st.set_page_config(
    page_title="Recruiter Candidate Finder",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Recruiter Talent Pool Matcher")
st.markdown("---")

# Sidebar for Recruiter Inputs
st.sidebar.header("Search Parameters")

# 📥 Bulk Upload Resumes
uploaded_resumes = st.sidebar.file_uploader(
    "📥 Bulk Upload Resumes", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True,
    help="Files uploaded here enrich the Talent Pool folder instantly."
)

# 📋 Upload Job Description
uploaded_jd = st.sidebar.file_uploader(
    "📋 Upload Job Description",
    type=["pdf", "docx", "txt"],
)

job_description_text = st.sidebar.text_area(
    "Or Paste Job Description", 
    height=200,
    placeholder="Paste the Job Description to find matching candidates..."
)

top_k = st.sidebar.number_input(
    "Target Candidates to Match", 
    min_value=1, 
    max_value=10, 
    value=3,
    step=1
)

search_btn = st.sidebar.button("🔬 Search Candidates", use_container_width=True)

if search_btn:
    if not uploaded_resumes:
        st.sidebar.error("⚠️ Please upload at least one Resume to analyze!")
        st.stop()

    # 1. Process Bulk Uploads first
    resumes_dir = os.path.join("data", "resumes")
    os.makedirs(resumes_dir, exist_ok=True)
    
    st.sidebar.info(f"Saving {len(uploaded_resumes)} resume(s)...")
    for res_file in uploaded_resumes:
        save_path = os.path.join(resumes_dir, res_file.name)
        with open(save_path, "wb") as f:
            f.write(res_file.getvalue())


    # 2. Resolve Job Description Input
    jd_content = ""
    if uploaded_jd:
        with st.spinner("Extracting Job Description file..."):
            temp_path = os.path.join("data", f"temp_jd_{uploaded_jd.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_jd.getvalue())
            jd_content = extract_text_from_pdf(temp_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        jd_content = job_description_text.strip()

    if not jd_content:
        st.sidebar.error("Please provide a Job Description via File or Text Area!")
    else:
        with st.spinner("Searching Talent Pool..."):
            
            # 1. Load from Directory (On-the-fly / Cache)
            cache_dir = os.path.join("data", "cache_parsed")
            os.makedirs(cache_dir, exist_ok=True)

            st.toast("Reading current candidates...", icon="⏳")
            # Filter solely to the sidebar uploaded queue titles
            files = [f.name for f in uploaded_resumes]


            candidate_results = []

            # 2. Process Candidates
            for file_name in files:
                st.toast(f"Processing candidate {file_name}...", icon="📄")
                file_path = os.path.join(resumes_dir, file_name)
                cache_path = os.path.join(cache_dir, f"{file_name}.json")
                
                structured_data = None
                
                # Check Cache
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            structured_data = json.load(f)
                    except Exception:
                        pass
                
                # If not cached or corrupt, parse via AI
                if not structured_data:
                    st.toast(f"  🧠 AI Parsing {file_name}...", icon="⏳")
                    text = extract_text_from_pdf(file_path)
                    if "Error" not in text:
                        structured_data = parse_resume(text)
                        # Save to cache
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(structured_data, f)

                if structured_data and "error" not in structured_data:
                    # 3. Match against Job Description
                    st.toast(f"  🔬 Matching {file_name}...", icon="✨")
                    # passing jd_content instead of previous job_description
                    match_data = match_resume_to_job(structured_data, jd_content)
                    
                    candidate_results.append({

                        "name": structured_data.get("contact_info", {}).get("name") or file_name,
                        "filename": file_name,
                        "profile": structured_data,
                        "match": match_data
                    })


            if not candidate_results:
                st.warning("No candidates matched your search criteria.")
                st.stop()

            # 4. Sort Results by Score Descending
            candidate_results.sort(key=lambda x: x["match"].get("match_score", 0), reverse=True)
            candidate_results = candidate_results[:int(top_k)]


            st.success(f"Found {len(candidate_results)} match(es) in the pool!")

            # --- DISPLAY RESULTS ---
            for rank, candidate in enumerate(candidate_results, 1):
                with st.container():
                    match = candidate["match"]
                    score = match.get("match_score", 0)
                    
                    st.markdown(f"### #{rank}: {candidate['name']} (`{candidate['filename']}`)")
                    
                    col_score, col_details = st.columns([1, 4])
                    
                    with col_score:
                        st.metric(label="📊 Match Score", value=f"{score}%")
                        
                        # Granular Breakdown Display
                        breakdown = match.get("score_breakdown", {})
                        if breakdown:
                            st.markdown("---")
                            st.caption("🔍 Breakdown")
                            st.progress(min(1.0, breakdown.get('skills', 0) / 40), text=f"🛠️ Skills: {breakdown.get('skills', 0)}/40")
                            st.progress(min(1.0, breakdown.get('experience', 0) / 30), text=f"💼 Exp: {breakdown.get('experience', 0)}/30")
                            st.progress(min(1.0, breakdown.get('projects', 0) / 20), text=f"💡 Proj: {breakdown.get('projects', 0)}/20")
                            st.progress(min(1.0, breakdown.get('education', 0) / 10), text=f"🎓 Edu: {breakdown.get('education', 0)}/10")

                    
                    with col_details:
                        st.subheader("💡 Key Strengths")
                        st.write(match.get("strengths") or "Not analyzed.")
                        
                        st.subheader("⚠️ Missing Core Keywords")
                        missing = match.get("missing_keywords", [])
                        st.write(", ".join(missing) if missing else "Perfect Keyword Overlap!")

                        st.subheader("🔧 Recommendation to secure interview")
                        tailoring = match.get("tailoring_advice", [])
                        for advice in tailoring:
                            st.markdown(f"- {advice}")
                    
                    with st.expander("👁‍🗨 View Candidate Profile"):
                        # Profile details
                        prof = candidate["profile"]
                        col_ext1, col_ext2 = st.columns(2)
                        with col_ext1:
                            st.write("**Contact:**")
                            cont = prof.get("contact_info", {})
                            st.markdown(f"- Email: {cont.get('email')}")
                            st.markdown(f"- Phone: {cont.get('phone')}")
                            st.markdown(f"- Location: {cont.get('location')}")
                        with col_ext2:
                            st.write("**Core Skills:**")
                            st.write(", ".join(prof.get("skills", [])))
                        
                        st.write("**Experience:**")
                        for exp in prof.get("experience", []):
                            st.markdown(f"**{exp.get('title')}** @ {exp.get('company')} ({exp.get('duration')})")
                            for bullet in exp.get("achievements", []):
                                st.markdown(f"  - {bullet}")

                        if prof.get("projects"):
                            st.write("**Projects:**")
                            for proj in prof.get("projects", []):
                                st.markdown(f"**{proj.get('title')}**")
                                for highlight in proj.get("highlights", []):
                                    st.markdown(f"  - {highlight}")

                    st.markdown("---")


