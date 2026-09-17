"""
AI Resume & Job Matching Platform
==================================
Upload a resume and paste a job description. The app:
1. Extracts the resume's text
2. Computes a match score using embeddings-based semantic similarity
   (not just keyword overlap -- it understands that "led a team" and
   "managed staff" mean similar things)
3. Uses an LLM to identify missing keywords/skills and give concrete,
   actionable suggestions to improve the resume for this specific job

Run with:
    streamlit run app.py
"""

import json
import re

import streamlit as st
import streamlit.components.v1 as components
import cohere
from pypdf import PdfReader
import numpy as np


# =============================================================================
# CONFIG
# =============================================================================

EMBEDDING_MODEL = "embed-english-v3.0"
GENERATION_MODEL = "command-a-plus-05-2026"

st.set_page_config(
    page_title="AI Resume Matcher",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLING -- slate + coral "modern career tool" theme
# =============================================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f8fa;
    }
    .block-container {
        max-width: 1300px;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li {
        color: #232830;
    }
    h1, h2, h3 { color: #1a1e24 !important; }

    /* Sidebar (drawer) -- deep slate with coral accents */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1e24 0%, #262b33 100%);
        border-right: 1px solid rgba(255, 111, 89, 0.2);
    }
    section[data-testid="stSidebar"] * {
        color: #e8eaed !important;
    }
    section[data-testid="stSidebar"] h1 {
        border: 1px solid rgba(255, 111, 89, 0.45) !important;
        background: rgba(255, 111, 89, 0.10);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 0.6rem;
    }
    section[data-testid="stSidebar"] h3 {
        display: inline-block;
        border: 1px solid rgba(255, 111, 89, 0.5) !important;
        background: rgba(255, 111, 89, 0.12);
        color: #ff9478 !important;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Bordered cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border-radius: 16px !important;
        border: 1px solid #e2e5ea !important;
        border-top: 3px solid #ff6f59 !important;
        box-shadow: 0 3px 16px rgba(26, 30, 36, 0.06);
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.6rem;
    }
    @media (max-width: 640px) {
        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.85rem 1rem;
        }
    }

    /* Sidebar collapse/expand toggle arrow */
    .st-emotion-cache-12bp31y {
        color: #ff6f59 !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] svg,
    button[kind="header"] svg {
        color: #ff6f59 !important;
        fill: #ff6f59 !important;
    }

    /* Direct fix for Streamlit's internal upload button class (consistent across apps on the same Streamlit version) */
    .st-emotion-cache-1uufcrr {
        background: transparent !important;
        border: 1px solid #ff6f59 !important;
        color: #ff6f59 !important;
    }
    .st-emotion-cache-1uufcrr * {
        color: #ff6f59 !important;
        fill: #ff6f59 !important;
    }

    /* File uploader */
    div[data-testid="stFileUploaderDropzone"] {
        background: #fbfbfc !important;
        border: 2px dashed #ff6f59 !important;
        border-radius: 14px !important;
    }
    div[data-testid="stFileUploaderDropzone"] * {
        color: #4a5058 !important;
    }
    div[data-testid="stFileUploaderDropzone"] [role="button"],
    div[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] label,
    div[data-testid="stBaseButton-secondary"] {
        background: transparent !important;
        border: 1px solid #ff6f59 !important;
        color: #ff6f59 !important;
    }
    div[data-testid="stFileUploaderDropzone"] [role="button"] *,
    div[data-testid="stFileUploaderDropzone"] button *,
    div[data-testid="stFileUploaderDropzone"] label *,
    div[data-testid="stBaseButton-secondary"] * {
        color: #ff6f59 !important;
        fill: #ff6f59 !important;
    }

    /* Inputs */
    div[data-baseweb="input"], div[data-baseweb="textarea"], textarea {
        border-radius: 10px !important;
        border: 1px solid #d8dce2 !important;
        background: #ffffff !important;
    }
    div[data-baseweb="input"] input, textarea {
        color: #1a1e24 !important;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
        border-color: #ff6f59 !important;
        box-shadow: 0 0 0 3px rgba(255, 111, 89, 0.15);
    }

    /* Buttons */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 700;
        background: #ff6f59;
        color: #ffffff !important;
        border: none;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #e85a44;
    }

    /* Result cards */
    .keyword-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        background: #fee2e2;
        border: 1px solid #fca5a5;
        color: #b91c1c;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px 4px 3px 0;
    }
    .keyword-chip.matched {
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #15803d;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2e5ea !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HEADER + SIDEBAR
# =============================================================================

def display_header():
    components.html(
        """
        <style>
            html, body { margin: 0; padding: 0; background: transparent; }
            .hero-box {
                font-family: 'Source Sans Pro', sans-serif;
                background: linear-gradient(135deg, #1a1e24 0%, #262b33 100%);
                border: 1px solid rgba(255, 111, 89, 0.35);
                border-left: 6px solid #ff6f59;
                border-radius: 18px;
                padding: 1.5rem 1.7rem;
                box-sizing: border-box;
            }
            .hero-title {
                margin: 0 0 0.4rem 0;
                font-size: 2.1rem;
                font-weight: 800;
                color: #ff9478;
            }
            .hero-subtitle {
                color: #c7cbd1;
                margin: 0 0 0.9rem 0;
                font-size: 0.98rem;
                line-height: 1.5;
                max-width: 640px;
            }
            .badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
            .badge {
                display: inline-block;
                padding: 4px 13px;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.4px;
                border: 1px solid rgba(255,111,89,0.5);
                background: rgba(255,111,89,0.14);
                color: #ff9478;
            }
            @media (max-width: 480px) {
                .hero-box { padding: 1.1rem 1.2rem; }
                .hero-title { font-size: 1.5rem; margin-bottom: 0.3rem; }
                .hero-subtitle { font-size: 0.85rem; margin-bottom: 0.7rem; }
                .badge { font-size: 0.65rem; padding: 3px 10px; }
            }
        </style>
        <div class="hero-box">
            <h1 class="hero-title">AI Resume Matcher</h1>
            <p class="hero-subtitle">
                Upload your resume and a job description to get a semantic match score,
                missing keywords, and concrete suggestions to improve your chances.
            </p>
            <div class="badge-row">
                <span class="badge">Semantic Matching</span>
                <span class="badge">Keyword Analysis</span>
                <span class="badge">AI Suggestions</span>
            </div>
        </div>
        """,
        height=210,
    )


def display_sidebar():
    with st.sidebar:
        st.title("AI Resume Matcher")
        st.caption(
            "Compares your resume against a job description using semantic "
            "embeddings, then uses AI to suggest concrete improvements."
        )

        st.subheader("How it works")
        st.write("1. Upload your resume (PDF)")
        st.write("2. Paste the job description")
        st.write("3. Get a match score (0-100)")
        st.write("4. See missing keywords + suggestions")

        st.subheader("Why Semantic Matching")
        st.write(
            "Unlike simple keyword search, this compares *meaning* -- "
            "'led a team of 5' will match 'people management experience' "
            "even without shared exact words."
        )

        st.subheader("Tech")
        st.write("Python")
        st.write("Streamlit")
        st.write("Cohere API (embeddings + chat)")
        st.write("pypdf")


# =============================================================================
# API KEY SETUP
# =============================================================================

def get_api_key():
    """Get the Cohere API key from Streamlit secrets, or ask the user for it."""
    try:
        if "COHERE_API_KEY" in st.secrets:
            return st.secrets["COHERE_API_KEY"]
    except Exception:
        pass
    return st.sidebar.text_input("Enter your Cohere API key", type="password")


# =============================================================================
# PDF PROCESSING
# =============================================================================

def extract_text_from_pdf(uploaded_file) -> str:
    """Read all text out of an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


# =============================================================================
# MATCHING
# =============================================================================

def compute_match_score(client, resume_text: str, job_description: str) -> float:
    """Compute a semantic similarity score (0-100) between the resume and
    job description using embeddings, rather than simple keyword overlap."""
    result = client.embed(
        model=EMBEDDING_MODEL,
        texts=[resume_text, job_description],
        input_type="search_document",
        embedding_types=["float"],
    )
    vecs = np.array(result.embeddings.float_)
    resume_vec, job_vec = vecs[0], vecs[1]

    cosine_sim = np.dot(resume_vec, job_vec) / (np.linalg.norm(resume_vec) * np.linalg.norm(job_vec))
    # Cosine similarity for text embeddings tends to sit in a narrow band
    # (rarely below ~0.2 or above ~0.9) -- rescale to a more intuitive 0-100.
    score = np.clip((cosine_sim - 0.15) / (0.75 - 0.15), 0, 1) * 100
    return round(float(score), 1)


def extract_text(message_content) -> str:
    """Find the text content item in a message's content list. Newer
    Cohere models can include a 'thinking' block before the actual text
    answer, so we can't assume content[0] is always the text."""
    for item in message_content:
        if hasattr(item, "text"):
            return item.text
    return ""


def analyze_resume_gap(client, resume_text: str, job_description: str) -> dict:
    """Ask the LLM to identify missing keywords/skills and give concrete
    suggestions for tailoring the resume to this job description."""
    prompt = f"""You are a resume optimization assistant. Compare the resume below against
the job description, and respond with ONLY a JSON object (no other text) in this exact format:

{{
  "matched_keywords": ["keyword1", "keyword2", ...],
  "missing_keywords": ["keyword1", "keyword2", ...],
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
  "experience_gap_warning": null or "a short sentence"
}}

- matched_keywords: important skills/terms from the job description that ARE reflected in the resume (even if worded differently)
- missing_keywords: important skills/terms from the job description that are NOT reflected in the resume at all
- suggestions: 3-5 concrete, specific ways to improve the resume for this job. Each suggestion must be
  about better presenting, clarifying, quantifying, or tailoring experience that ALREADY EXISTS in the
  resume. NEVER tell the user to invent, add, or create a new project, job, certification, skill,
  deployment, or achievement they haven't demonstrated. NEVER give vague advice like "improve your
  resume" or "add more skills". If a suggestion involves a skill that might not be present, phrase it
  conditionally (e.g. "If you have experience with AI agents, highlight it in your Projects section" --
  NOT "Add a new AI agent project").
- experience_gap_warning: set to null UNLESS the job description explicitly requires a specific amount of
  prior professional experience (e.g. "3+ years", "senior", "mid-level", "substantial professional
  experience") AND the resume does not show comparable professional experience (e.g. it's a
  student/fresher resume, or only shows internships/coursework/personal projects with no matching
  professional role). If both conditions are true, set this to a short sentence like: "Experience
  requirement: This role requires 3+ years of experience, while your resume does not show comparable
  professional experience." Do not invent employment history. If the job doesn't state an experience
  requirement, or the resume already shows relevant professional experience, set this to null.

Job Description:
{job_description}

Resume:
{resume_text}

JSON:"""

    response = client.chat(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = extract_text(response.message.content) if response.message.content else "{}"

    # Strip markdown code fences if the model wrapped the JSON in them
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "matched_keywords": [],
            "missing_keywords": [],
            "suggestions": ["The AI's response couldn't be parsed. Please try again."],
            "experience_gap_warning": None,
        }


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    display_sidebar()
    display_header()

    api_key = get_api_key()
    if not api_key:
        st.info("Enter your Cohere API key in the sidebar to get started.")
        st.stop()

    client = cohere.ClientV2(api_key=api_key)

    left_col, right_col = st.columns([1, 1.3], gap="large")

    with left_col:
        with st.container(border=True):
            st.subheader("Your Resume")
            resume_file = st.file_uploader("Upload resume PDF", type="pdf", label_visibility="collapsed")

            st.subheader("Job Description")
            job_description = st.text_area(
                "Job description",
                label_visibility="collapsed",
                height=220,
                placeholder="Paste the full job description here...",
            )

            analyze_clicked = st.button("Analyze Match", type="primary", use_container_width=True)

    if analyze_clicked:
        if not resume_file:
            st.error("Please upload a resume PDF first.")
            st.stop()
        if not job_description.strip():
            st.error("Please paste a job description first.")
            st.stop()
        if len(job_description.split()) < 30:
            st.warning(
                "Job description is too short for a reliable match. "
                "Add the full job description for a more accurate result. "
                "Proceeding anyway -- treat this result as a rough estimate."
            )

        with st.spinner("Reading your resume..."):
            resume_text = extract_text_from_pdf(resume_file)
            if not resume_text.strip():
                st.error("Couldn't extract any text from this PDF. It might be a scanned image PDF.")
                st.stop()

        with st.spinner("Computing match score..."):
            try:
                score = compute_match_score(client, resume_text, job_description)
            except Exception as exc:
                st.error(f"Something went wrong computing the match score: {exc}")
                st.stop()

        with st.spinner("Analyzing keywords and generating suggestions..."):
            try:
                analysis = analyze_resume_gap(client, resume_text, job_description)
            except Exception as exc:
                st.error(f"Something went wrong analyzing the resume: {exc}")
                st.stop()

        st.session_state["result"] = {
            "score": score,
            "analysis": analysis,
        }

    with right_col:
        if "result" not in st.session_state:
            with st.container(border=True):
                st.subheader("Results")
                st.caption("Upload a resume and job description, then click Analyze Match to see your results here.")
        else:
            result = st.session_state["result"]
            score = result["score"]
            analysis = result["analysis"]
            matched = analysis.get("matched_keywords", [])
            missing = analysis.get("missing_keywords", [])

            with st.container(border=True):
                st.caption("MATCH SCORE")
                st.metric(label="", value=f"{score:.0f} / 100", label_visibility="collapsed")
                if score >= 75:
                    st.success("Strong match for this role.")
                elif score >= 50:
                    st.warning("Moderate match -- some tailoring recommended.")
                else:
                    st.error("Low match -- significant tailoring needed.")

                total_keywords = len(matched) + len(missing)
                if total_keywords > 0:
                    skills_match_pct = round(len(matched) / total_keywords * 100)
                    st.divider()
                    st.metric("Skills Match", f"{skills_match_pct}%")

                experience_warning = analysis.get("experience_gap_warning")
                if experience_warning:
                    st.warning(experience_warning)

            with st.container(border=True):
                st.markdown("**Matched Keywords**")
                if matched:
                    chips = "".join(f'<span class="keyword-chip matched">{kw}</span>' for kw in matched)
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("None identified.")

                st.markdown("**Missing Keywords**")
                if missing:
                    chips = "".join(f'<span class="keyword-chip">{kw}</span>' for kw in missing)
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("No missing keywords identified.")

            with st.container(border=True):
                st.markdown("**Suggestions to Improve Your Resume**")
                for i, suggestion in enumerate(analysis.get("suggestions", []), 1):
                    st.markdown(f"{i}. {suggestion}")


if __name__ == "__main__":
    main()