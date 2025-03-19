import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import docx
import os
import re
from job_matcher import process_selected_resumes

def extract_text_from_pdf(file):
    text = ""
    try:
        with fitz.open(stream=file.getvalue(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except fitz.EmptyFileError:
        st.error(f"Error: {file.name} is an empty or invalid PDF.")
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_name_email(text):
    """Extracts name and email from resume text."""
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    email = email_match.group(0) if email_match else "Unknown"
    name = text.split("\n")[0] if text else "Unknown"
    return name.strip(), email.strip()

def remove_duplicates(resume_files):
    """Ensures only one resume per candidate is kept, preferring PDFs."""
    unique_candidates = {}
    filtered_files = []
    
    for resume in resume_files:
        text = extract_text_from_pdf(resume) if resume.type == "application/pdf" else extract_text_from_docx(resume)
        name, email = extract_name_email(text)
        
        key = (name, email)
        if key not in unique_candidates or resume.type == "application/pdf":
            unique_candidates[key] = resume
    
    return list(unique_candidates.values())

def save_shortlisted_resumes(shortlisted_resumes, resume_files):
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "shortlisted resumes")
    os.makedirs(desktop_path, exist_ok=True)
    
    for resume in resume_files:
        if resume.name in shortlisted_resumes:
            resume_path = os.path.join(desktop_path, resume.name)
            with open(resume_path, "wb") as f:
                f.write(resume.getvalue())  # Read actual file content and save

def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: left;'>Resume Score Calculator</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload Resumes")
        resume_files = st.file_uploader("", type=["pdf", "docx"], accept_multiple_files=True)
        st.markdown("### Upload Job Description")
        jd_file = st.file_uploader("", type=["pdf", "docx"], accept_multiple_files=False)
        calculate = st.button("Calculate")
    
    if "results" not in st.session_state:
        st.session_state.results = None
    
    with col2:
        if calculate:
            if not resume_files or not jd_file:
                st.warning("Please upload both resumes and a job description file to proceed.")
            else:
                resume_files = remove_duplicates(resume_files)  # Remove duplicates before processing
                
                jd_text = extract_text_from_pdf(jd_file) if jd_file.type == "application/pdf" else extract_text_from_docx(jd_file)
                resume_texts = {
                    resume.name: extract_text_from_pdf(resume) if resume.type == "application/pdf" else extract_text_from_docx(resume)
                    for resume in resume_files
                }
                
                st.session_state.results = process_selected_resumes(jd_text, resume_texts)
        
        if st.session_state.results:
            df = pd.DataFrame(st.session_state.results, columns=["Resume", "Score (%)"])
            
            # Display all results
            st.markdown("### All Results")
            st.dataframe(df, use_container_width=True)
            
            # Show score filter input after scores are displayed
            score_threshold = st.number_input("Enter Minimum Similarity Score (%)", min_value=0, max_value=100, value=50)
            
            # Filter based on threshold
            filtered_df = df[df["Score (%)"] >= score_threshold].reset_index(drop=True)
            filtered_df.index += 1  # Add serial numbering starting from 1
            st.markdown("### Filtered Results")
            st.dataframe(filtered_df, use_container_width=True)
            
            if not filtered_df.empty:
                if st.button("Shortlist"):
                    save_shortlisted_resumes(filtered_df["Resume"].tolist(), resume_files)
                    st.success("Shortlisted resumes saved on desktop.")
                    st.session_state.filtered_df = filtered_df
            
            if "filtered_df" in st.session_state:
                st.markdown("### Filtered Results (After Shortlisting)")
                st.dataframe(st.session_state.filtered_df, use_container_width=True)

if __name__ == "__main__":
    main()
