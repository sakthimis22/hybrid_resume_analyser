import pandas as pd
import re
import os

# Resolve paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load resume dataset
resumes = pd.read_csv(os.path.join(PROJECT_ROOT, "datasets", "raw", "resumes", "Resume", "Resume.csv"))

# Load skills dataset (needs latin-1 encoding)
skills = pd.read_csv(
    os.path.join(PROJECT_ROOT, "datasets", "raw", "skills", "IT_Job_Roles_Skills.csv"),
    encoding="latin-1"
)
# Skills column contains comma-separated lists per row — split and flatten
all_skills = []
for entry in skills['Skills'].dropna().tolist():
    for skill in str(entry).split(','):
        cleaned = skill.strip().lower()
        if cleaned:
            all_skills.append(cleaned)
skill_list = list(set(all_skills))
print(f"Loaded {len(skill_list)} unique skills for matching.")


def extract_skills(text):
    """Extract matching skills from resume text."""
    text = text.lower()
    found_skills = []

    for skill in skill_list:
        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))


def extract_experience(text):
    """Extract years of experience from resume text."""
    exp = re.findall(r'\d+\s+years', text.lower())
    return exp[0] if exp else "0 years"


# Apply extraction
resumes['skills_extracted'] = resumes['Resume_str'].apply(extract_skills)
resumes['experience'] = resumes['Resume_str'].apply(extract_experience)

# Ensure output directory exists
output_path = os.path.join(PROJECT_ROOT, "datasets", "processed", "resume_entities.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save structured resume data
resumes.to_csv(output_path, index=False)

print("Resume entity extraction complete.")
print(f"Output saved to: {output_path}")
print(f"Total resumes processed: {len(resumes)}")
print(f"\nSample output:")
print(resumes[['ID', 'skills_extracted', 'experience', 'Category']].head())
