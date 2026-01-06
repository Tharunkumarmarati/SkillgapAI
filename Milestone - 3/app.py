from flask import Flask, render_template, request
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.io as pio
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

JD_SKILLS = [
    "Python",
    "SQL",
    "Machine Learning",
    "AWS Cloud Services",
    "Data Analysis",
    "Leadership"
]

SKILL_VOCAB = [
    "python",
    "sql",
    "machine learning",
    "data analysis",
    "aws",
    "statistics",
    "communication",
    "leadership"
]

@app.route("/", methods=["GET", "POST"])
def index():
    matrix_html = None
    donut_html = None
    matched, partial, missing = [], [], []
    overall = 0

    if request.method == "POST":
        file = request.files["resume"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Extract resume text
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text().lower()

        # Extract skills
        resume_skills = [s.title() for s in SKILL_VOCAB if s in text]
        if not resume_skills:
            resume_skills = ["No Relevant Skills"]

        # Similarity calculation
        resume_emb = model.encode(resume_skills)
        jd_emb = model.encode(JD_SKILLS)
        similarity = cosine_similarity(resume_emb, jd_emb)

        # =========================
        # 🔵 Bubble Similarity Matrix
        # =========================
        x_vals, y_vals, sizes, colors = [], [], [], []

        for i, r_skill in enumerate(resume_skills):
            for j, jd_skill in enumerate(JD_SKILLS):
                score = similarity[i][j]
                x_vals.append(jd_skill)
                y_vals.append(r_skill)

                sizes.append(score * 40 + 10)

                if score >= 0.8:
                    colors.append("#2ecc71")   # Green
                elif score >= 0.5:
                    colors.append("#f39c12")   # Orange
                else:
                    colors.append("#e74c3c")   # Red

        fig = go.Figure(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.85
            ),
            hovertemplate=
                "Resume Skill: %{y}<br>"
                "JD Skill: %{x}<br>"
                "Similarity Score: %{marker.size:.2f}<extra></extra>"
        ))

        fig.update_layout(
            title="Similarity Matrix",
            xaxis_title="Job Description Skills",
            yaxis_title="Resume Skills",
            height=350,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        matrix_html = pio.to_html(fig, full_html=False)

        # =========================
        # Skill Classification
        # =========================
        for j, skill in enumerate(JD_SKILLS):
            score = similarity[:, j].max()
            if score >= 0.8:
                matched.append(skill)
            elif score >= 0.5:
                partial.append(skill)
            else:
                missing.append(skill)

        overall = round((len(matched) / len(JD_SKILLS)) * 100)

        # =========================
        # Donut Chart
        # =========================
        donut_fig = go.Figure(go.Pie(
            labels=["Matched", "Partial", "Missing"],
            values=[len(matched), len(partial), len(missing)],
            hole=0.6,
            marker_colors=["#2ecc71", "#f39c12", "#e74c3c"]
        ))

        donut_fig.update_layout(
            height=260,
            margin=dict(t=20, b=20)
        )

        donut_html = pio.to_html(donut_fig, full_html=False)

    return render_template(
        "index.html",
        matrix=matrix_html,
        donut=donut_html,
        matched=matched,
        partial=partial,
        missing=missing,
        overall=overall,
        show=bool(matrix_html)
    )

if __name__ == "__main__":
    app.run(debug=True)
