import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PATHS ----------------
PROCESSED_DIR = "datasets/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RESUME_FILE = os.path.join(PROCESSED_DIR, "resumes_cleaned.csv")
JOB_FILE = os.path.join(PROCESSED_DIR, "jobs_cleaned.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# ---------------- LOAD DATA ----------------
resume_df = pd.read_csv(RESUME_FILE)
job_df = pd.read_csv(JOB_FILE)

# Use FIRST job description for now
job_text = job_df["clean_job_description"].iloc[0]

resume_texts = resume_df["clean_resume"].astype(str)

# ---------------- LOAD MODEL ----------------
tfidf = joblib.load(MODEL_FILE)

# ---------------- VECTORIZE ----------------
resume_vectors = tfidf.transform(resume_texts)
job_vector = tfidf.transform([job_text])

# ---------------- COSINE SIMILARITY ----------------
similarity_scores = cosine_similarity(resume_vectors, job_vector).flatten()

# ---------------- CREATE RESULT ----------------
results_df = pd.DataFrame({
    "Resume_ID": resume_df.index,
    "Similarity_Score": similarity_scores
})

results_df = results_df.sort_values(
    by="Similarity_Score",
    ascending=False
)

# ---------------- SAVE OUTPUT ----------------
output_path = os.path.join(OUTPUT_DIR, "ranked_candidates.csv")
results_df.to_csv(output_path, index=False)

print("✅ Candidate ranking completed successfully")
print(f"📁 Results saved at: {output_path}")
