import streamlit as st
import joblib
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import numpy as np

# Load model, vectorizer, and label map
model = joblib.load("hevoxnext_model.pkl")
tfidf = joblib.load("hevoxnext_vectorizer.pkl")
with open("hevoxnext_label_map.json", "r") as f:
    label_map = json.load(f)

# Define keyword flags
keyword_dict = {
    'has_agriculture': ['farm', 'crop', 'irrigation', 'pesticide', 'soil', 'harvest', 'cultivation'],
    'has_sales': ['sales', 'target', 'lead', 'revenue', 'client acquisition', 'closing', 'crm'],
    'has_chef': ['kitchen', 'cooking', 'culinary', 'menu', 'dish', 'chef', 'food preparation'],
    'has_pr': ['public relations', 'media', 'press release', 'branding', 'campaign', 'communication'],
    'has_consultant': ['consulting', 'advisory', 'strategy', 'analysis', 'stakeholders', 'client engagement']
}

# Text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create feature flags
def extract_flags(text):
    return [int(any(kw in text for kw in kws)) for kws in keyword_dict.values()]

# Streamlit UI
st.set_page_config(page_title="HevoxNEXT Resume Matcher", layout="centered")
st.title("🚀 HevoxNEXT Resume ↔ Job Description Matcher")

resume_input = st.text_area("Paste Resume Text Here", height=250)
jd_input = st.text_area("Paste Job Description Here", height=250)

if st.button("🔍 Match Now"):
    if not resume_input or not jd_input:
        st.warning("Please enter both Resume and Job Description.")
    else:
        cleaned_resume = clean_text(resume_input)
        cleaned_jd = clean_text(jd_input)

        # TF-IDF transform
        resume_vec = tfidf.transform([cleaned_resume])
        jd_vec = tfidf.transform([cleaned_jd])

        # Feature flags
        resume_flags = np.array(extract_flags(cleaned_resume)).reshape(1, -1)
        resume_combined = hstack([resume_vec, resume_flags])

        # Predict category
        pred_label = model.predict(resume_combined)[0]
        pred_category = label_map[str(pred_label)]

        # Cosine similarity
        similarity = cosine_similarity(resume_vec, jd_vec)[0][0] * 100

        # Keyword match
        resume_keywords = set(cleaned_resume.split())
        jd_keywords = set(cleaned_jd.split())
        common_keywords = resume_keywords & jd_keywords
        missing_keywords = jd_keywords - resume_keywords

        # Output
        st.success(f"🎯 Match Score: {similarity:.2f}%")
        st.info(f"📌 Predicted Category: {pred_category}")

        st.subheader("✅ Common Keywords")
        st.write(", ".join(list(common_keywords)[:15]))

        st.subheader("⚠️ Missing Keywords")
        st.write(", ".join(list(missing_keywords)[:15]))
