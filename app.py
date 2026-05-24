import gradio as gr
import PyPDF2
import docx

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Skills Database
# =========================
skills_db = [
    "python",
    "sql",
    "machine learning",
    "tensorflow",
    "pytorch",
    "langchain",
    "langgraph",
    "llamaindex",
    "fastapi",
    "streamlit",
    "gradio",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "nlp",
    "rag",
    "pinecone",
    "faiss"
]


# =========================
# Read PDF
# =========================
def read_pdf(path):

    text = ""

    with open(path, "rb") as file:

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# =========================
# Read DOCX
# =========================
def read_docx(path):

    doc = docx.Document(path)

    text = ""

    for para in doc.paragraphs:
        text += para.text + "\n"

    return text


# =========================
# Extract Skills
# =========================
def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills_db:

        if skill.lower() in text:
            found.append(skill)

    return found


# =========================
# ATS Score
# =========================
def ats_score(resume_text, jd):

    docs = [resume_text, jd]

    cv = CountVectorizer()

    matrix = cv.fit_transform(docs)

    score = cosine_similarity(matrix)[0][1]

    return round(score * 100, 2)


# =========================
# Main Function
# =========================
def analyze_resume(file, job_description):

    try:

        if file is None:
            return """
            <div style='color:red;
                        font-size:25px;
                        text-align:center;
                        padding:30px;'>
            Upload Resume First
            </div>
            """

        filepath = file

        if isinstance(file, dict):
            filepath = file["name"]

        elif not isinstance(file, str):
            filepath = file.name


        # Read Resume
        if filepath.endswith(".pdf"):

            resume_text = read_pdf(filepath)

        elif filepath.endswith(".docx"):

            resume_text = read_docx(filepath)

        else:
            return "<h1>Only PDF/DOCX Supported</h1>"


        # Skills
        skills = extract_skills(resume_text)

        # ATS Score
        score = ats_score(resume_text, job_description)

        # Missing Skills
        missing = []

        jd_lower = job_description.lower()

        for skill in skills_db:

            if skill in jd_lower and skill not in skills:
                missing.append(skill)


        # Score Color
        if score >= 75:
            score_color = "#00FF99"
            score_status = "Excellent Match"

        elif score >= 50:
            score_color = "#FFD700"
            score_status = "Good Match"

        else:
            score_color = "#FF4B4B"
            score_status = "Low Match"


        # Skill Cards
        skill_cards = ""

        for skill in skills:

            skill_cards += f"""
            <span style="
                background:#00C2FF;
                color:white;
                padding:10px 18px;
                border-radius:25px;
                margin:6px;
                display:inline-block;
                font-size:16px;
                font-weight:bold;
                box-shadow:0px 0px 10px rgba(0,194,255,0.6);
            ">
            {skill}
            </span>
            """


        # Missing Skill Cards
        missing_cards = ""

        for skill in missing:

            missing_cards += f"""
            <span style="
                background:#FF4B4B;
                color:white;
                padding:10px 18px;
                border-radius:25px;
                margin:6px;
                display:inline-block;
                font-size:16px;
                font-weight:bold;
                box-shadow:0px 0px 10px rgba(255,75,75,0.6);
            ">
            {skill}
            </span>
            """


        # Final Stylish HTML
        result = f"""

        <div style="
            background: linear-gradient(135deg, #0F172A, #111827, #1E293B);
            padding:40px;
            border-radius:25px;
            color:white;
            font-family:Arial;
            box-shadow:0px 0px 30px rgba(0,0,0,0.5);
        ">

        <div style="text-align:center;">

            <h1 style="
                color:#00FFCC;
                font-size:50px;
                margin-bottom:10px;
                text-shadow:0px 0px 20px #00FFCC;
            ">
            AI Resume Analyzer
            </h1>

            <p style="
                color:#CBD5E1;
                font-size:18px;
            ">
            Smart ATS + Skill Intelligence Dashboard
            </p>

        </div>

        <hr style="margin-top:30px;
                   margin-bottom:30px;
                   border:1px solid #334155;">


        <!-- ATS SCORE -->
        <div style="
            text-align:center;
            background:#111827;
            padding:35px;
            border-radius:20px;
            margin-bottom:35px;
            border:2px solid {score_color};
            box-shadow:0px 0px 25px {score_color};
        ">

            <h2 style="
                color:#E2E8F0;
                font-size:30px;
            ">
            ATS MATCH SCORE
            </h2>

            <h1 style="
                color:{score_color};
                font-size:90px;
                margin:10px;
                text-shadow:0px 0px 25px {score_color};
            ">
            {score}%
            </h1>

            <h3 style="
                color:{score_color};
                font-size:25px;
            ">
            {score_status}
            </h3>

        </div>


        <!-- DETECTED SKILLS -->
        <div style="
            background:#0F172A;
            padding:25px;
            border-radius:20px;
            margin-bottom:30px;
            border:1px solid #334155;
        ">

            <h2 style="
                color:#38BDF8;
                font-size:30px;
                margin-bottom:20px;
            ">
            Detected Skills
            </h2>

            {skill_cards}

        </div>


        <!-- MISSING SKILLS -->
        <div style="
            background:#0F172A;
            padding:25px;
            border-radius:20px;
            border:1px solid #334155;
        ">

            <h2 style="
                color:#FF6B6B;
                font-size:30px;
                margin-bottom:20px;
            ">
            Missing Skills
            </h2>

            {missing_cards}

        </div>

        </div>
        """

        return result

    except Exception as e:

        return f"""
        <div style='
            color:red;
            font-size:25px;
            padding:30px;
            text-align:center;
        '>
        ERROR: {str(e)}
        </div>
        """


# =========================
# GRADIO UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # AI Resume Analyzer
    
    Upload Resume and Compare with Job Description
    """)

    with gr.Row():

        file_input = gr.File(
            type="filepath",
            label="Upload Resume"
        )

        jd_input = gr.Textbox(
            lines=15,
            label="Job Description"
        )

    analyze_btn = gr.Button(
        "Analyze Resume",
        variant="primary"
    )

    output = gr.HTML()

    analyze_btn.click(
        fn=analyze_resume,
        inputs=[file_input, jd_input],
        outputs=output
    )

app.launch()
