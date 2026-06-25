import pandas as pd
import re
import os

# ---------------- PATHS (MATCH YOUR STRUCTURE) ----------------
RESUME_PATH = "datasets/raw/resumes/Resume/Resume.csv"
JOB_PATH = "datasets/raw/job_descriptions/job_descriptions.csv"
SKILL_PATH = "datasets/raw/skills/IT_Job_Roles_Skills.csv"

PROCESSED_DIR = "datasets/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------- TEXT CLEANING FUNCTION ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- RESUME DATASET ----------------
resume_df = pd.read_csv(RESUME_PATH)
resume_df = resume_df.dropna()

# Resume text column (safe detection)
resume_text_col = resume_df.columns[0]
resume_df["clean_resume"] = resume_df[resume_text_col].apply(clean_text)

resume_df = resume_df.drop_duplicates()
resume_df.to_csv(
    os.path.join(PROCESSED_DIR, "resumes_cleaned.csv"),
    index=False
)

print("✅ Resume dataset processed")

# ---------------- JOB DESCRIPTION DATASET ----------------
job_df = pd.read_csv(JOB_PATH)
job_df = job_df.dropna()

job_text_col = job_df.columns[1]
job_df["clean_job_description"] = job_df[job_text_col].apply(clean_text)

job_df = job_df.drop_duplicates()
job_df.to_csv(
    os.path.join(PROCESSED_DIR, "jobs_cleaned.csv"),
    index=False
)

print("✅ Job description dataset processed")

# ---------------- SKILLS DATASET ----------------
# ---------------- SKILLS DATASET (ENCODING SAFE) ----------------
try:
    skills_df = pd.read_csv(SKILL_PATH, encoding="utf-8")
except UnicodeDecodeError:
    skills_df = pd.read_csv(SKILL_PATH, encoding="latin1")

skills_df = skills_df.dropna()

skills_text_col = skills_df.columns[1]
skills_df["clean_skills"] = skills_df[skills_text_col].apply(clean_text)

skills_df = skills_df.drop_duplicates()
skills_df.to_csv(
    os.path.join(PROCESSED_DIR, "skills_cleaned.csv"),
    index=False
)

print("✅ Skills dataset processed (encoding handled)")


print("\n🎉 STEP 2 COMPLETED SUCCESSFULLY")
