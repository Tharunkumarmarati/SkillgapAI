from flask import Flask, render_template, request
import os
import pdfplumber
import docx
import spacy
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# Load NLP Models
# -----------------------------
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Skill Dictionaries
# -----------------------------
TECHNICAL_SKILLS = [
    "python", "machine learning", "sql", "data analysis",
    "java", "tensorflow", "power bi", "excel"
]

SOFT_SKILLS = [
    "communication", "teamwork", "leadership",
    "problem solving", "time management"
]

# -----------------------------
# Text Extraction
# -----------------------------
def extract_text(file_path):
    text = ""

    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    return text.lower()

# -----------------------------
# spaCy Skill Detection
# -----------------------------
def spacy_skill_candidates(text, skill_list):
    found = set()
    text = text.replace("-", " ")

    for skill in skill_list:
        if skill in text:
            found.add(skill)

    return list(found)

# -----------------------------
# BERT Semantic Validation
# -----------------------------
def bert_score_skills(text, skills):
    scored = []
    text_embedding = bert_model.encode(text[:1000], convert_to_tensor=True)

    for skill in skills:
        skill_embedding = bert_model.encode(skill, convert_to_tensor=True)
        score = util.cos_sim(skill_embedding, text_embedding).item()
        scored.append((skill.title(), round(score * 100, 2)))

    return scored


# -----------------------------
# Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    tech_skills = []
    soft_skills = []

    if request.method == "POST":
        file = request.files.get("resume")

        if file and file.filename != "":
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            resume_text = extract_text(file_path)

            # spaCy → candidates
            tech_candidates = spacy_skill_candidates(resume_text, TECHNICAL_SKILLS)
            soft_candidates = spacy_skill_candidates(resume_text, SOFT_SKILLS)

            # BERT → validation
            tech_skills = bert_score_skills(resume_text, tech_candidates)
            soft_skills = [(skill.title(), 100) for skill in soft_candidates]


    return render_template(
        "index.html",
        tech_skills=tech_skills,
        soft_skills=soft_skills
    )

if __name__ == "__main__":
    app.run(debug=True)

