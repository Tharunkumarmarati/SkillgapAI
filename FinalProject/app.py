import streamlit as st
import pandas as pd
import numpy as np
import spacy
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os  # Added for folder management

# --- PAGE CONFIG ---
st.set_page_config(page_title="SkillGapAI", layout="wide")

# --- FILE STORAGE SETUP ---
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def save_uploaded_file(uploaded_file):
    """Saves the uploaded file to the local uploads directory."""
    try:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# --- CUSTOM CSS FOR DASHBOARD UI ---
st.markdown("""
    <style>
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
    .skill-tag { background-color: #e8f0fe; color: #1a73e8; padding: 5px 10px; border-radius: 15px; margin: 2px; display: inline-block; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_nlp_models():
    nlp = spacy.load("en_core_web_sm")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, bert_model

nlp, bert_model = load_nlp_models()

# --- MILESTONE 1: PARSING FUNCTIONS ---
def extract_text_from_pdf(file):
    # Use seek(0) to ensure we read from the start if saved previously
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def extract_text_from_docx(file):
    file.seek(0)
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_document(uploaded_file):
    if uploaded_file is None: return ""
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf': return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx': return extract_text_from_docx(uploaded_file)
    elif ext == 'txt': 
        uploaded_file.seek(0)
        return str(uploaded_file.read(), "utf-8")
    return ""

# --- MILESTONE 2: UPDATED EXTRACTION (Categorized) ---
def get_classified_skills(text):
    tech_skills = ["Python", "Machine Learning", "TensorFlow", "SQL", "NoSQL", "AWS", "Tableau", "Power BI", "Statistics", "Adv. Stats"]
    soft_skills = ["Communication", "Project Mgmt", "Team Leadership", "Problem Solving", "Adaptability"]
    found_tech = [s for s in tech_skills if s.lower() in text.lower()]
    found_soft = [s for s in soft_skills if s.lower() in text.lower()]
    return list(set(found_tech)), list(set(found_soft))

# --- APP LAYOUT ---
st.title("🚀 SkillGapAI: Advanced Analysis")

# SIDEBAR: MILESTONE 1 - UPLOADS
st.sidebar.header("Data Ingestion 📁")
res_file = st.sidebar.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])
jd_file = st.sidebar.file_uploader("Upload Job Description", type=['pdf', 'docx', 'txt'])

if res_file and jd_file:
    # SAVE TO UPLOADS FOLDER
    res_path = save_uploaded_file(res_file)
    jd_path = save_uploaded_file(jd_file)
    
    if res_path and jd_path:
        st.sidebar.success(f"Files saved to `{UPLOAD_DIR}/` folder")

    # Parsing
    resume_text = parse_document(res_file)
    jd_text = parse_document(jd_file)

    # --- MILESTONE 1: PREVIEW ---
    with st.expander("📄 Milestone 1: Document Preview", expanded=False):
        c1, c2 = st.columns(2)
        c1.subheader("Resume Preview")
        c1.text_area("Parsed Content", resume_text[:1000], height=150, key="res_p")
        c2.subheader("JD Preview")
        c2.text_area("Parsed Content", jd_text[:1000], height=150, key="jd_p")

    # --- MILESTONE 2: EXTRACTION & DONUT CHART ---
    res_tech, res_soft = get_classified_skills(resume_text)
    jd_tech, jd_soft = get_classified_skills(jd_text)
    res_skills = res_tech + res_soft
    jd_skills = jd_tech + jd_soft

    st.header("🔍 Milestone 2: Skill Extraction")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Technical Skills", len(res_tech))
    m_col2.metric("Soft Skills", len(res_soft))
    m_col3.metric("Total Skills", len(res_skills))

    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        st.write("**Skill Tag Distribution**")
        donut_fig = px.pie(names=["Technical Skills", "Soft Skills"], values=[len(res_tech), len(res_soft)], hole=0.5, color_discrete_sequence=["#1a73e8", "#34a853"])
        st.plotly_chart(donut_fig, use_container_width=True)
    with col_e2:
        st.write("**Extracted Resume Skills:**")
        st.write(" ".join([f'<span class="skill-tag">{s}</span>' for s in res_skills]), unsafe_allow_html=True)
        st.write("**Job Requirements:**")
        st.write(" ".join([f'<span class="skill-tag">{s}</span>' for s in jd_skills]), unsafe_allow_html=True)

    # --- MILESTONE 3: SIMILARITY MATCHING ---
    st.header("📊 Milestone 3: Similarity Matching")
    if res_skills and jd_skills:
        res_embs = bert_model.encode(res_skills)
        jd_embs = bert_model.encode(jd_skills)
        cos_sim = util.cos_sim(res_embs, jd_embs).numpy()

        matrix_data = []
        for i, r_s in enumerate(res_skills):
            for j, j_s in enumerate(jd_skills):
                score = float(cos_sim[i][j])
                cat = "High Match" if score >= 0.8 else "Partial Match" if score >= 0.5 else "Low Match"
                matrix_data.append({"Resume Skill": r_s, "Job Skill": j_s, "Score": score, "Category": cat})
        
        df_matrix = pd.DataFrame(matrix_data)
        best_matches = df_matrix.groupby("Job Skill")["Score"].max().reset_index()
        high_c = len(best_matches[best_matches['Score'] >= 0.8])
        part_c = len(best_matches[(best_matches['Score'] >= 0.5) & (best_matches['Score'] < 0.8)])
        low_c = len(best_matches[best_matches['Score'] < 0.5])

        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Highly Matched", high_c)
        c_m2.metric("Partially Matched", part_c)
        c_m3.metric("Low Matched", low_c)
        
        fig_sim = px.scatter(df_matrix, x="Resume Skill", y="Job Skill", size="Score", color="Category", color_discrete_map={"High Match": "#2ecc71", "Partial Match": "#f1c40f", "Low Match": "#e74c3c"}, title="BERT Cosine Similarity Matrix")
        st.plotly_chart(fig_sim, use_container_width=True)

        # --- MILESTONE 4: DASHBOARD ---
        st.divider()
        st.header("📈 Milestone 4: Skill Match Overview")
        avg_score = best_matches["Score"].mean()
        st.metric("Overall Match Percentage", f"{avg_score*100:.1f}%")

        comparison_df = pd.DataFrame({"Skill": best_matches["Job Skill"], "Resume Score": best_matches["Score"] * 100, "JD Requirement": [100] * len(best_matches)}).melt(id_vars="Skill", var_name="Type", value_name="Percentage")
        fig_side = px.bar(comparison_df, x="Skill", y="Percentage", color="Type", barmode="group", color_discrete_sequence=["#1a73e8", "#bdc3c7"])
        st.plotly_chart(fig_side, use_container_width=True)

        missing_skills_list = best_matches[best_matches['Score'] < 0.5]["Job Skill"].tolist()
        st.subheader("⚠️ Missing Skills Identified")
        if missing_skills_list:
            for s in missing_skills_list: st.error(f"❌ **{s}** - No significant match found in resume.")
        else: st.success("✅ All required skills matched!")

        cat_radar = ["Technical", "Soft Skills", "Tools", "Experience", "Leadership"]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=[80, 70, 90, 60, 50], theta=cat_radar, fill='toself', name='Resume'))
        fig_radar.add_trace(go.Scatterpolar(r=[90, 85, 80, 80, 70], theta=cat_radar, fill='toself', name='Job Requirements'))
        st.plotly_chart(fig_radar, use_container_width=True)

        csv = df_matrix.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Analysis Report (CSV)", data=csv, file_name="analysis.csv", mime="text/csv")
else:
    st.info("Please upload documents in the sidebar to begin your final analysis.")