import nltk
# import fitz  # PyMuPDF for PDF processing
from ats import ATS  # Import the ATS class
import concurrent.futures
import logging

# Ensure NLTK resources are loaded
def ensure_nltk_resources_loaded():
    resources = ['stopwords', 'punkt', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

ensure_nltk_resources_loaded()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_resume(jd_text, resume_filename, resume_text):
    ats = ATS()
    ats.load_job_description(jd_text)
    ats.load_resume(resume_text)
    try:
        experience = ats.extract_experience()
        ats.clean_experience(experience)
        skills = " ".join(ats.extract_skills())
        ats.clean_skills(skills)
        similarity_score = ats.compute_similarity()
    except AttributeError as e:
        logging.error(f"Error processing resume {resume_filename}: {e}")
        return resume_filename, 0  # Return zero score in case of error
    return resume_filename, round(similarity_score * 100, 2)

def process_selected_resumes(jd_text, resume_texts):
    """Process only the selected resumes."""
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_resume, jd_text, resume_filename, resume_text) 
                   for resume_filename, resume_text in resume_texts.items()]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"Error processing resume: {e}")
    
    return results
