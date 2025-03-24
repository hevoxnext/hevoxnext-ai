import streamlit as st
import pandas as pd
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import docx
from PyPDF2 import PdfReader

# Load model and vectorizer from saved files
@st.cache_resource
def load_model():
    model = pickle.load(open('hevoxnext_model.pkl', 'rb'))
    vectorizer = pickle.load(open('hevoxnext_vectorizer.pkl', 'rb'))
    with open('hevoxnext_label_map.json') as f:
        label_map = json.load(f)
    return model, vectorizer, label_map

# Function to process the text and predict the match score
def predict_match_score(resume_text, jd_text, model, vectorizer):
    resume_vec = vectorizer.transform([resume_text])
    jd_vec = vectorizer.transform([jd_text])
    
    # Predicting the match score using cosine similarity
    cosine_sim = cosine_similarity(resume_vec, jd_vec)
    match_score = cosine_sim[0][0]
    return match_score

# Streamlit UI setup
st.title("HevoxNEXT - Resume Matcher")
st.write("Upload your Resume and Job Description (JD) to match and optimize.")

# Upload files
resume_file = st.file_uploader("Upload Resume (Text file)", type=["txt", "pdf", "docx"])
jd_file = st.file_uploader("Upload Job Description (Text file)", type=["txt", "pdf", "docx"])


if resume_file and jd_file:
    # Read and extract text from uploaded resume and job description
    resume_text = extract_text_from_file(resume_file)
    jd_text = extract_text_from_file(jd_file)
    
    # Load the model and vectorizer
    model, vectorizer, label_map = load_model()
    
    # Get match score
    match_score = predict_match_score(resume_text, jd_text, model, vectorizer)
    
    # Display match score
    st.write(f"Match Score: {match_score * 100:.2f}%")
    
    # Provide feedback based on the match score
    if match_score < 0.5:
        st.warning("Your resume may need optimization for better ATS compatibility.")
    else:
        st.success("Your resume is well matched with the job description.")

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        if not text:
            st.warning("No text extracted from the PDF. The file may contain scanned images or complex formatting.")
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "
"
        return text
    return ""
