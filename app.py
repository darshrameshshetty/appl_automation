import os
import requests
import pdfplumber
import streamlit as st

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fpdf import FPDF

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Job Application AI Agent",
    page_icon="💼",
    layout="wide"
)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
)


# ============================================================
# 2. LANGGRAPH STATE
# ============================================================

class JobApplicationState(TypedDict):

    cv_text: str
    job_description: str

    cv_summary: str
    cover_letter: str
    interview_questions: str


# ============================================================
# 3. CV PDF TEXT EXTRACTION
# ============================================================

def extract_cv_text(uploaded_file):

    text = ""

    with pdfplumber.open(uploaded_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# ============================================================
# 4. JOB SCRAPER
# ============================================================

def scrape_job(url):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Remove unnecessary HTML
    for tag in soup(
        ["script", "style", "nav", "footer", "header"]
    ):
        tag.decompose()

    text = soup.get_text(
        separator="\n"
    )

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


# ============================================================
# 5. NODE 1 - TAILORED CV SUMMARY
# ============================================================

def generate_cv_summary(state):

    prompt = ChatPromptTemplate.from_template("""

You are an expert resume writer.

Candidate CV:
{cv_text}

Job Description:
{job_description}

Create a tailored professional summary for this candidate.

Rules:

1. Use ONLY information present in the CV.
2. Highlight skills relevant to the job.
3. Do not invent experience.
4. Do not invent companies.
5. Do not invent achievements.
6. Keep the summary concise and professional.
7. Match the candidate's existing professional tone.

Return only the final summary.

""")

    messages = prompt.format_messages(
        cv_text=state["cv_text"],
        job_description=state["job_description"]
    )

    response = llm.invoke(messages)

    return {
        "cv_summary": response.content
    }


# ============================================================
# 6. NODE 2 - COVER LETTER
# ============================================================

def generate_cover_letter(state):

    prompt = ChatPromptTemplate.from_template("""

You are an expert job application writer.

Candidate CV:
{cv_text}

Job Description:
{job_description}

Write a personalized cover letter.

Rules:

1. Use only information from the CV.
2. Connect the candidate's skills to the job.
3. Match the writing style of the CV.
4. Do not invent experience.
5. Do not invent achievements.
6. Do not claim technologies that are not present in the CV.
7. Keep the letter professional and concise.

Return only the cover letter.

""")

    messages = prompt.format_messages(
        cv_text=state["cv_text"],
        job_description=state["job_description"]
    )

    response = llm.invoke(messages)

    return {
        "cover_letter": response.content
    }


# ============================================================
# 7. NODE 3 - INTERVIEW QUESTIONS
# ============================================================

def generate_interview_questions(state):

    prompt = ChatPromptTemplate.from_template("""

You are an expert technical interviewer.

Candidate CV:
{cv_text}

Job Description:
{job_description}

Generate the TOP 3 interview questions that this candidate
is likely to be asked for this particular job.

For each question provide:

Question:
Why this question is likely:
Suggested Answer:

Rules:

1. Base the answers only on the CV.
2. Do not invent experience.
3. Questions should be specific to the job.
4. Include a mixture of technical and project-related questions
   when appropriate.

""")

    messages = prompt.format_messages(
        cv_text=state["cv_text"],
        job_description=state["job_description"]
    )

    response = llm.invoke(messages)

    return {
        "interview_questions": response.content
    }


# ============================================================
# 8. BUILD LANGGRAPH
# ============================================================

def create_graph():

    builder = StateGraph(JobApplicationState)

    # Add nodes
    builder.add_node(
        "cv_summary",
        generate_cv_summary
    )

    builder.add_node(
        "cover_letter",
        generate_cover_letter
    )

    builder.add_node(
        "interview_questions",
        generate_interview_questions
    )

    # START → three independent nodes
    builder.add_edge(
        START,
        "cv_summary"
    )

    builder.add_edge(
        START,
        "cover_letter"
    )

    builder.add_edge(
        START,
        "interview_questions"
    )

    # Nodes → END
    builder.add_edge(
        "cv_summary",
        END
    )

    builder.add_edge(
        "cover_letter",
        END
    )

    builder.add_edge(
        "interview_questions",
        END
    )

    return builder.compile()


graph = create_graph()


# ============================================================
# 9. PDF GENERATION
# ============================================================

def create_pdf(
    cv_summary,
    cover_letter,
    interview_questions
):

    pdf = FPDF()

    pdf.add_page()

    # Register Unicode font
    pdf.add_font(
        "DejaVu",
        "",
        "DejaVuSans.ttf"
    )

    pdf.add_font(
        "DejaVu",
        "B",
        "DejaVuSans.ttf"
    )

    # ========================================================
    # TITLE
    # ========================================================

    pdf.set_font(
        "DejaVu",
        "B",
        18
    )

    pdf.cell(
        0,
        12,
        "Job Application Report",
        ln=True,
        align="C"
    )

    pdf.ln(10)

    # ========================================================
    # CV SUMMARY
    # ========================================================

    pdf.set_font(
        "DejaVu",
        "B",
        14
    )

    pdf.cell(
        0,
        10,
        "Tailored CV Summary",
        ln=True
    )

    pdf.set_font(
        "DejaVu",
        "",
        11
    )

    pdf.multi_cell(
        0,
        7,
        cv_summary
    )

    pdf.ln(8)

    # ========================================================
    # COVER LETTER
    # ========================================================

    pdf.set_font(
        "DejaVu",
        "B",
        14
    )

    pdf.cell(
        0,
        10,
        "Personalized Cover Letter",
        ln=True
    )

    pdf.set_font(
        "DejaVu",
        "",
        11
    )

    pdf.multi_cell(
        0,
        7,
        cover_letter
    )

    pdf.ln(8)

    # ========================================================
    # INTERVIEW QUESTIONS
    # ========================================================

    pdf.set_font(
        "DejaVu",
        "B",
        14
    )

    pdf.cell(
        0,
        10,
        "Likely Interview Questions",
        ln=True
    )

    pdf.set_font(
        "DejaVu",
        "",
        11
    )

    pdf.multi_cell(
        0,
        7,
        interview_questions
    )

    # ========================================================
    # SAVE PDF
    # ========================================================

    output_file = "job_application_report.pdf"

    pdf.output(output_file)

    return output_file


# ============================================================
# 10. STREAMLIT UI
# ============================================================

st.title("💼 Job Application Automation Agent")

st.write(
    "Upload your CV and provide a job posting URL. "
    "The LangGraph agent will generate tailored application materials."
)


# ============================================================
# CV UPLOAD
# ============================================================

uploaded_cv = st.file_uploader(
    "Upload your CV",
    type=["pdf"]
)


# ============================================================
# JOB URL
# ============================================================

job_url = st.text_input(
    "Enter Job Posting URL"
)


# ============================================================
# GENERATE BUTTON
# ============================================================

if st.button(
    "🚀 Generate Application",
    type="primary"
):

    if uploaded_cv is None:

        st.error(
            "Please upload your CV."
        )

        st.stop()

    if not job_url:

        st.error(
            "Please enter a job posting URL."
        )

        st.stop()


    # --------------------------------------------------------
    # STEP 1: EXTRACT CV
    # --------------------------------------------------------

    with st.spinner("Reading CV..."):

        try:

            cv_text = extract_cv_text(
                uploaded_cv
            )

        except Exception as e:

            st.error(
                f"Could not read CV: {e}"
            )

            st.stop()


    # --------------------------------------------------------
    # STEP 2: SCRAPE JOB
    # --------------------------------------------------------

    with st.spinner("Scraping job description..."):

        try:

            job_description = scrape_job(
                job_url
            )

        except Exception as e:

            st.error(
                f"Could not scrape job posting: {e}"
            )

            st.stop()


    # --------------------------------------------------------
    # STEP 3: RUN LANGGRAPH
    # --------------------------------------------------------

    with st.spinner(
        "Generating personalized application..."
    ):

        try:

            result = graph.invoke({

                "cv_text": cv_text,

                "job_description":
                    job_description,

                "cv_summary": "",

                "cover_letter": "",

                "interview_questions": ""
            })

        except Exception as e:

            st.error(
                f"Agent failed: {e}"
            )

            st.stop()


    # Save result in session
    st.session_state["cv_summary"] = (
        result["cv_summary"]
    )

    st.session_state["cover_letter"] = (
        result["cover_letter"]
    )

    st.session_state["interview_questions"] = (
        result["interview_questions"]
    )

    st.success(
        "Application generated successfully!"
    )


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

if "cv_summary" in st.session_state:

    st.subheader("Generated Application")

    tab1, tab2, tab3 = st.tabs([
        "📄 CV Summary",
        "✉️ Cover Letter",
        "🎯 Interview Questions"
    ])


    # ========================================================
    # TAB 1
    # ========================================================

    with tab1:

        st.session_state["cv_summary"] = st.text_area(

            "Edit your CV Summary",

            value=st.session_state["cv_summary"],

            height=250
        )


    # ========================================================
    # TAB 2
    # ========================================================

    with tab2:

        st.session_state["cover_letter"] = st.text_area(

            "Edit your Cover Letter",

            value=st.session_state["cover_letter"],

            height=400
        )


    # ========================================================
    # TAB 3
    # ========================================================

    with tab3:

        st.session_state["interview_questions"] = st.text_area(

            "Edit Interview Questions and Answers",

            value=st.session_state[
                "interview_questions"
            ],

            height=500
        )


    # ========================================================
    # PDF EXPORT
    # ========================================================

    st.divider()

    if st.button("📥 Export PDF"):

        pdf_file = create_pdf(

            st.session_state["cv_summary"],

            st.session_state["cover_letter"],

            st.session_state[
                "interview_questions"
            ]
        )

        with open(
            pdf_file,
            "rb"
        ) as file:

            st.download_button(

                label="Download PDF",

                data=file,

                file_name="job_application_report.pdf",

                mime="application/pdf"
            )