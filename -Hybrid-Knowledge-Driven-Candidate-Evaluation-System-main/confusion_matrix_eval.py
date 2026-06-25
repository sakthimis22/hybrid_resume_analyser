import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- PATHS ----------------
RESUME_FILE = "datasets/processed/resumes_cleaned.csv"
JOB_FILE = "datasets/processed/jobs_cleaned.csv"
MODEL_FILE = "models/tfidf_vectorizer.pkl"

# ---------------- LOAD DATA ----------------
resume_df = pd.read_csv(RESUME_FILE)
job_df = pd.read_csv(JOB_FILE)

# DEBUG: print columns
print("Resume columns:", resume_df.columns.tolist())
print("Job columns:", job_df.columns.tolist())

resume_texts = resume_df["clean_resume"].astype(str)
resume_categories = resume_df.iloc[:, 1]   # usually Category column

job_text = job_df["clean_job_description"].iloc[0]

# ---------------- LOAD MODEL ----------------
tfidf = joblib.load(MODEL_FILE)

resume_vectors = tfidf.transform(resume_texts)
job_vector = tfidf.transform([job_text])

# ---------------- SIMILARITY ----------------
similarity_scores = cosine_similarity(resume_vectors, job_vector).flatten()

print("\nSample similarity scores:", similarity_scores[:10])

# ---------------- THRESHOLD (VERY IMPORTANT) ----------------
THRESHOLD = 0.05   # start LOW for debugging

y_pred = (similarity_scores >= THRESHOLD).astype(int)

# ---------------- GROUND TRUTH ----------------
# Assume resumes matching the job category are positives
target_category = resume_categories.mode()[0]
y_true = (resume_categories == target_category).astype(int)

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

print("\nCONFUSION MATRIX")
print(cm)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_true, y_pred))
