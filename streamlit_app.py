import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, util

# Load models and tools
model = joblib.load("hevoxnext_model.pkl")
vectorizer = joblib.load("hevoxnext_vectorizer.pkl")
with open("hevoxnext_label_map.json", "r") as f:
    label_map = eval(f.read())

# Page layout
st.set_page_config(page_title="HevoxNEXT | Resume Matcher", layout="centered")
st.title("📄 HevoxNEXT Resume - JD Matcher")
st.markdown("Match your resume with any job description using AI!")

# Upload resume
resume_text = st.text_area("✍️ Paste your Resume Text", height=250)

# Upload JD
jd_text = st.text_area("📄 Paste Job Description Text", height=250)

if st.button("🚀 Match Now") and resume_text and jd_text:
    with st.spinner("Analyzing..."):
        resume_clean = resume_text.lower()
        jd_clean = jd_text.lower()

        resume_vec = vectorizer.transform([resume_clean])
        pred_class = model.predict(resume_vec)[0]
        category = label_map.get(pred_class, "Unknown")

        transformer = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = transformer.encode([resume_text, jd_text], convert_to_tensor=True)
        semantic_score = float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0][0]) * 100

        st.success(f"🎯 Match Score: {semantic_score:.2f}%")
        st.info(f"📌 Predicted Category: {category}")
