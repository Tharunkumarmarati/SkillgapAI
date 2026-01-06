import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Resume Skill Matcher", layout="wide")

# ---------------- SKILL VOCABULARY ----------------
SKILL_SET = [
    "python", "java", "machine learning", "deep learning",
    "data analysis", "sql", "tensorflow", "pandas", "numpy",
    "data visualization", "power bi", "tableau", "excel",
    "statistics", "scikit-learn", "nlp", "aws", "docker"
]

# ---------------- SESSION STATE ----------------
for key, default in {
    "resume_text": "",
    "job_desc_text": "",
    "skill_table": None,
    "matched_skills": [],
    "missing_skills": [],
    "match_percentage": 0,
    "similarity_score": 0.0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- FUNCTIONS ----------------
def read_file(file):
    try:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            return " ".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            return "\n".join(p.text for p in doc.paragraphs)

        elif file.type == "text/plain":
            return file.getvalue().decode("utf-8")

        else:
            return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Upload Files", "Preview", "Skill Analysis", "Download Report"]
)

# ---------------- MAIN TITLE ----------------
st.title("📄 Resume vs Job Description Skill Matcher")
st.write("Analyze skill alignment using TF-IDF and cosine similarity.")

# ===================== UPLOAD PAGE =====================
if page == "Upload Files":
    st.subheader("📤 Upload Resume and Job Description")

    col1, col2 = st.columns(2)

    with col1:
        resume_file = st.file_uploader(
            "Upload Resume (PDF / DOCX / TXT)",
            type=["pdf", "docx", "txt"]
        )

    with col2:
        jd_file = st.file_uploader(
            "Upload Job Description (PDF / DOCX / TXT)",
            type=["pdf", "docx", "txt"]
        )

    if resume_file:
        st.session_state.resume_text = read_file(resume_file)

    if jd_file:
        st.session_state.job_desc_text = read_file(jd_file)

    if st.session_state.resume_text and st.session_state.job_desc_text:
        st.success("Files uploaded successfully!")

# ===================== PREVIEW PAGE =====================
elif page == "Preview":
    st.subheader("📌 Content Preview")

    if not st.session_state.resume_text or not st.session_state.job_desc_text:
        st.warning("Please upload both files first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.text_area("Resume Preview", st.session_state.resume_text, height=300)
        with c2:
            st.text_area("Job Description Preview", st.session_state.job_desc_text, height=300)

# ===================== SKILL ANALYSIS PAGE =====================
elif page == "Skill Analysis":
    st.subheader("🧠 Skill Analysis")

    if not st.session_state.resume_text or not st.session_state.job_desc_text:
        st.warning("Upload resume and job description first.")
    else:
        if st.button("Analyze Skills", key="analyze_btn"):

            resume_text = st.session_state.resume_text.lower()
            jd_text = st.session_state.job_desc_text.lower()

            vectorizer = TfidfVectorizer(vocabulary=SKILL_SET)
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

            resume_vector = tfidf_matrix[0].toarray()[0]
            jd_vector = tfidf_matrix[1].toarray()[0]

            resume_skills = [SKILL_SET[i] for i, v in enumerate(resume_vector) if v > 0]
            jd_skills = [SKILL_SET[i] for i, v in enumerate(jd_vector) if v > 0]

            matched = sorted(set(resume_skills) & set(jd_skills))
            missing = sorted(set(jd_skills) - set(resume_skills))

            st.session_state.matched_skills = matched
            st.session_state.missing_skills = missing
            st.session_state.match_percentage = (
                int(len(matched) / len(jd_skills) * 100) if jd_skills else 0
            )
            st.session_state.similarity_score = cosine_similarity(
                tfidf_matrix[0], tfidf_matrix[1]
            )[0][0]

            st.session_state.skill_table = pd.DataFrame({
                "Skill": jd_skills,
                "Status": ["Matched" if s in matched else "Missing" for s in jd_skills]
            })

        if st.session_state.skill_table is not None:
            m1, m2 = st.columns(2)
            m1.metric("Skill Match %", f"{st.session_state.match_percentage}%")
            m2.metric(
                "Resume–JD Similarity",
                f"{round(st.session_state.similarity_score * 100, 2)}%"
            )

            c1, c2 = st.columns(2)
            c1.success(", ".join(st.session_state.matched_skills) or "No matched skills")
            c2.error(", ".join(st.session_state.missing_skills) or "No missing skills")

            chart_df = pd.DataFrame({
                "Category": ["Matched", "Missing"],
                "Count": [
                    len(st.session_state.matched_skills),
                    len(st.session_state.missing_skills)
                ]
            })
            st.bar_chart(chart_df.set_index("Category"))

            st.dataframe(st.session_state.skill_table)

# ===================== DOWNLOAD PAGE =====================
elif page == "Download Report":
    st.subheader("📥 Download Skill Gap Report")

    if st.session_state.skill_table is None:
        st.warning("Run skill analysis first.")
    else:
        csv = st.session_state.skill_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="skill_gap_report.csv",
            mime="text/csv"
        )
