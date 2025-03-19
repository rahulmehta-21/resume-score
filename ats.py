import re
import spacy
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TextCleaner:
    """
    A class used to clean text by removing stopwords, punctuation, and performing lemmatization.
    """
    def __init__(self) -> None:
        self.set_of_stopwords = set(stopwords.words("english") + list(string.punctuation))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, raw_text: str) -> str:
        tokens = word_tokenize(raw_text.lower())
        tokens = [token for token in tokens if token not in self.set_of_stopwords]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        cleaned_text = " ".join(tokens)
        return cleaned_text

class ATS:
    """
    A class to parse resumes and job descriptions, extract relevant information,
    and compute similarities between resumes and job descriptions.
    """
    RESUME_SECTIONS = [
        "Contact Information", "Objective", "Summary", "Education", "Experience", 
        "Skills", "Projects", "Certifications", "Licenses", "Awards", "Honors", 
        "Publications", "References", "Technical Skills", "Computer Skills", 
        "Programming Languages", "Software Skills", "Soft Skills", "Language Skills", 
        "Professional Skills", "Transferable Skills", "Work Experience", 
        "Professional Experience", "Employment History", "Internship Experience", 
        "Volunteer Experience", "Leadership Experience", "Research Experience", 
        "Teaching Experience",
    ]

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_resume(self, resume_content):
        self.resume_content = resume_content

    def load_job_description(self, jd_content):
        self.jd_content = jd_content

    def extract_experience(self):
        """
        Extracts the work experience section from the resume content.
        """
        experience_start = self.resume_content.lower().find("experience")
        if experience_start == -1:
            return ""

        experience_end = len(self.resume_content)
        for section in self.RESUME_SECTIONS:
            section_start = self.resume_content.lower().find(section.lower(), experience_start + 1)
            if section_start != -1:
                experience_end = min(experience_end, section_start)
        
        experience_section = self.resume_content[experience_start:experience_end].strip()
        return experience_section

    def extract_skills(self):
        """
        Extracts skills from the resume content.
        """
        skills_pattern = re.compile(r'Skills\s*[:\n]', re.IGNORECASE)
        skills_match = skills_pattern.search(self.resume_content)
        
        if skills_match:
            skills_start = skills_match.end()
            skills_end = self.resume_content.find('\n\n', skills_start)
            skills_section = self.resume_content[skills_start:skills_end].strip()
            skills_lines = skills_section.split('\n')
            
            extracted_skills = []
            for line in skills_lines:
                line_skills = re.split(r'[:,-]', line)
                extracted_skills.extend([skill.strip() for skill in line_skills if skill.strip()])
            
            return list(set(extracted_skills))
        else:
            return []

    def clean_experience(self, experience):
        cleaner = TextCleaner()
        self.cleaned_experience = cleaner.clean_text(experience)

    def clean_skills(self, skills):
        cleaner = TextCleaner()
        self.cleaned_skills = cleaner.clean_text(skills)

    def clean_jd(self):
        cleaner = TextCleaner()
        return cleaner.clean_text(self.jd_content)

    def compute_similarity(self):
        cleaned_resume = self.cleaned_experience + " " + self.cleaned_skills
        cleaned_jd_text = self.clean_jd()
        
        resume_embedding = self.model.encode([cleaned_resume])
        jd_embedding = self.model.encode([cleaned_jd_text])
        
        similarity_score = cosine_similarity(resume_embedding, jd_embedding)[0][0]
        return similarity_score

def main():
    resume_content = input("\n\nPlease enter the resume content: ")
    jd_content = input("\n\nPlease enter the job description content: ")
    
    ats = ATS()
    ats.load_resume(resume_content)
    ats.load_job_description(jd_content)
    
    experience = ats.extract_experience()
    ats.clean_experience(experience)
    
    skills = " ".join(ats.extract_skills())
    ats.clean_skills(skills)
    
    similarity_score = ats.compute_similarity()
    print(f"The similarity score between the resume and job description is: {round(similarity_score * 100, 2)}%")

if __name__ == "__main__":
    main()
