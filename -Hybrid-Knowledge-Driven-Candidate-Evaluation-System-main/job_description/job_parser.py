import pandas as pd
import os

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Load skills dataset (using the common file)
try:
    skills_df = pd.read_csv(
        os.path.join(PROJECT_ROOT, "datasets", "raw", "skills", "IT_Job_Roles_Skills.csv"), 
        encoding="latin-1"
    )
    all_skills = []
    for entry in skills_df['Skills'].dropna().tolist():
        for skill in str(entry).split(','):
            cleaned = skill.strip().lower()
            if cleaned:
                all_skills.append(cleaned)
    skill_list = list(set(all_skills))
except Exception:
    # Fallback to a basic list if file is missing
    skill_list = ["python", "machine learning", "tensorflow", "docker", "kubernetes", "react", "java", "sql"]


def extract_job_skills(job_text):
    """
    Extracts IT skills automatically from a pasted raw job description 
    by matching against the known knowledge base.
    """
    job_text = job_text.lower()
    found_skills = []

    for skill in skill_list:
        if skill in job_text:
            found_skills.append(skill)

    return list(set(found_skills))
