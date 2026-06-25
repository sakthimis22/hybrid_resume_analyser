"""
resume_ranking_pipeline.py

Takes the dataset of uploaded real-world resumes (PDFs/DOCXs) and
ranks them against the target job description using the ML Similarity Engine.

Output: outputs/uploaded_resume_rankings.csv
"""

import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

INPUT_RESUMES = os.path.join(PROJECT_ROOT, "outputs", "uploaded_resume_texts.csv")
RAW_JOB_FILE = os.path.join(PROJECT_ROOT, "datasets", "raw", "job_descriptions", "job_descriptions.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "uploaded_resume_rankings.csv")


def run_ranking_pipeline(job_text_override=None):
    
    print("Loading uploaded resumes...")
    try:
        resumes = pd.read_csv(INPUT_RESUMES)
    except FileNotFoundError:
        print(f"Error: {INPUT_RESUMES} not found. Run multi_resume_analyzer.py first.")
        return
        
    if resumes.empty:
        print("Error: No resumes to process.")
        return

    if job_text_override:
        print("Using provided high-signal job text...")
        job_text = job_text_override.lower()
    else:
        print("Loading target job description...")
        try:
            jobs_df = pd.read_csv(RAW_JOB_FILE, encoding="utf-8")
        except UnicodeDecodeError:
            jobs_df = pd.read_csv(RAW_JOB_FILE, encoding="latin-1")
        except FileNotFoundError:
            print(f"Error: Job description file not found at {RAW_JOB_FILE}")
            return
            
        # We use the raw job description, combining skills + responsibilities
        raw_skills = str(jobs_df["skills"].iloc[0]) if "skills" in jobs_df.columns else ""
        raw_resp = str(jobs_df["Responsibilities"].iloc[0]) if "Responsibilities" in jobs_df.columns else ""
        job_text = (raw_skills + " " + raw_resp).lower()
    
    # We dynamically fit the TF-IDF so we don't rely on the stale saved model 
    # (as discovered during the rank_candidates.py fix)
    print("Vectorizing text and computing ML similarity...")
    documents = [job_text] + resumes["resume_text"].fillna("").tolist()
    
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(documents)
    
    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    
    # Compute similarity between Job(0) and Resumes(1:)
    scores = cosine_similarity(resume_vectors, job_vector).flatten()
    
    resumes["ML_Score"] = scores
    
    # Sort candidates
    resumes = resumes.sort_values(by="ML_Score", ascending=False)
    
    # Add Rank
    resumes.insert(0, "Rank", range(1, len(resumes) + 1))
    
    resumes.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'=' * 60}")
    print("✅ Resume Ranking Pipeline Completed!")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"{'=' * 60}")
    print("\nTop Uploaded Candidates:")
    
    for _, r in resumes.head(5).iterrows():
        print(f"  #{r['Rank']}  {r['file_name']:<20} Score: {r['ML_Score']:.4f}")


if __name__ == "__main__":
    run_ranking_pipeline()
