import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- PATHS ----------------
PROCESSED_DIR = "datasets/processed"
MODEL_DIR = "models"

RESUME_FILE = os.path.join(PROCESSED_DIR, "resumes_cleaned.csv")
JOB_FILE = os.path.join(PROCESSED_DIR, "jobs_cleaned.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
resume_df = pd.read_csv(RESUME_FILE)
job_df = pd.read_csv(JOB_FILE)

resume_text = resume_df["clean_resume"].astype(str)
job_text = job_df["clean_job_description"].astype(str)

# ---------------- TRAIN TF-IDF ----------------
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

# Fit on BOTH resumes and job descriptions
combined_text = pd.concat([resume_text, job_text], axis=0)

tfidf.fit(combined_text)

# ---------------- SAVE MODEL ----------------
model_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
joblib.dump(tfidf, model_path)

print("✅ TF-IDF model trained and saved successfully")
print(f"📁 Saved at: {model_path}")
