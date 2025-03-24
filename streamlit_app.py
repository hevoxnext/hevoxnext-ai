import streamlit as st
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("hevoxnext_model.pkl")
vectorizer = joblib.load("hevoxnext_vectorizer.pkl")
with open("hevoxnext_label_map.json", "r") as f:
    label_map = json.load(f)

st.title("HevoxNEXT Resume-JD Matcher")
resume_input = st.text_area("Paste your Resume text here:")
jd_input = st.text_area("Paste the Job Description text here:")

if st.button("Match"):
    if resume_input and jd_input:
        tfidf_input = vectorizer.transform([resume_input])
        prediction = model.predict(tfidf_input)[0]
        st.markdown(f"🎯 **Predicted Category**: `{label_map[str(prediction)]}`")
    else:
        st.warning("Please provide both Resume and Job Description.")