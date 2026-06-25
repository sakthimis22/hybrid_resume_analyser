import pandas as pd
import joblib
import os
import re
import ast
from sklearn.metrics.pairwise import cosine_similarity

# ================================================================
# HYBRID SCORING ENGINE
# ================================================================
# Final Score = 0.6 × ML_similarity + 0.3 × Skill_match + 0.1 × Experience_match
# ================================================================

# ---------------- PATHS ----------------
PROCESSED_DIR = "datasets/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

RESUME_FILE = os.path.join(PROCESSED_DIR, "resumes_cleaned.csv")
JOB_FILE = os.path.join(PROCESSED_DIR, "jobs_cleaned.csv")
RAW_JOB_FILE = os.path.join("datasets", "raw", "job_descriptions", "job_descriptions.csv")
ENTITIES_FILE = os.path.join(PROCESSED_DIR, "resume_entities.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- SCORING WEIGHTS ----------------
WEIGHT_ML = 0.6
WEIGHT_SKILL = 0.3
WEIGHT_EXP = 0.1

# ---------------- LOAD DATA ----------------
print("Loading data...")
resume_df = pd.read_csv(RESUME_FILE)
job_df = pd.read_csv(JOB_FILE)
entities_df = pd.read_csv(ENTITIES_FILE)

# Load raw job data for actual description content
# (the cleaned version only has the experience range)
try:
    raw_job_df = pd.read_csv(RAW_JOB_FILE, encoding="utf-8")
except UnicodeDecodeError:
    raw_job_df = pd.read_csv(RAW_JOB_FILE, encoding="latin-1")

# Clean text function (matching preprocess_data.py logic)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Use actual resume text (Resume_str), not the broken clean_resume column
resume_texts = resume_df["Resume_str"].apply(clean_text)

# Build a rich job description from raw skills + responsibilities
raw_skills = str(raw_job_df["skills"].iloc[0]) if "skills" in raw_job_df.columns else ""
raw_resp = str(raw_job_df["Responsibilities"].iloc[0]) if "Responsibilities" in raw_job_df.columns else ""
job_text = re.sub(r'[^a-z0-9\s]', ' ', (raw_skills + " " + raw_resp).lower()).strip()

print(f"Job description length: {len(job_text)} chars")
print(f"Job skills preview: {raw_skills[:100]}...")

# Parse the job's required experience range (e.g., "5 to 15 Years")
job_experience_raw = str(job_df["Experience"].iloc[0]).lower()
exp_nums = re.findall(r'\d+', job_experience_raw)
if len(exp_nums) >= 2:
    job_exp_min = int(exp_nums[0])
    job_exp_max = int(exp_nums[1])
elif len(exp_nums) == 1:
    job_exp_min = int(exp_nums[0])
    job_exp_max = job_exp_min
else:
    job_exp_min = 0
    job_exp_max = 99

print(f"Job experience requirement: {job_exp_min}–{job_exp_max} years")

# ================================================================
# 1️⃣  ML SIMILARITY SCORE  (TF-IDF + Cosine)
# ================================================================
print("Computing ML similarity scores...")

# Fit TF-IDF on resumes + real job description for accurate similarity
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
combined = pd.concat([resume_texts, pd.Series([job_text])], ignore_index=True)
tfidf.fit(combined)

resume_vectors = tfidf.transform(resume_texts)
job_vector = tfidf.transform([job_text])
ml_scores = cosine_similarity(resume_vectors, job_vector).flatten()

# ================================================================
# 2️⃣  SKILL OVERLAP SCORE  (Knowledge-Driven with Inference)
# ================================================================
print("Computing inference-based skill overlap scores...")

# Import the inference-powered skill matcher
from ranking_engine.skill_matcher import skill_match_score as kb_skill_match, skill_kb

# Extract job skills by matching taxonomy skills against the job description
# (raw CSV skills are freeform phrases, so we use the same taxonomy-based approach)
job_text_lower = job_text.lower()
job_skills_list = []
for category, skills in skill_kb.items():
    for skill in skills:
        if skill.lower() in job_text_lower:
            job_skills_list.append(skill.lower())
job_skills_list = list(set(job_skills_list))
print(f"  Job skills extracted via taxonomy ({len(job_skills_list)}): {job_skills_list[:10]}...")


def compute_skill_score(skills_str):
    """Use inference-based matching for each resume's extracted skills."""
    try:
        resume_skills = ast.literal_eval(skills_str) if isinstance(skills_str, str) else []
    except (ValueError, SyntaxError):
        resume_skills = []

    if not resume_skills:
        return 0.0

    return kb_skill_match(resume_skills, job_skills_list)


skill_scores = entities_df["skills_extracted"].apply(compute_skill_score).values

# ================================================================
# 3️⃣  EXPERIENCE MATCH SCORE
# ================================================================
print("Computing experience match scores...")


def experience_match_score(exp_str):
    """Score how well candidate experience fits the job requirement range."""
    exp_nums = re.findall(r'\d+', str(exp_str))
    candidate_years = int(exp_nums[0]) if exp_nums else 0

    if job_exp_min <= candidate_years <= job_exp_max:
        return 1.0  # Perfect match — within range
    elif candidate_years < job_exp_min:
        # Under-qualified — partial credit based on how close
        return max(0, candidate_years / job_exp_min) if job_exp_min > 0 else 0.0
    else:
        # Over-qualified — slight penalty but still valuable
        return max(0.5, 1.0 - (candidate_years - job_exp_max) / 20)


exp_scores = entities_df["experience"].apply(experience_match_score).values

# ================================================================
# HYBRID SCORING — Combine all factors
# ================================================================
print(f"\nApplying hybrid scoring formula:")
print(f"  Final = {WEIGHT_ML}×ML + {WEIGHT_SKILL}×Skill + {WEIGHT_EXP}×Experience")

final_scores = (
    WEIGHT_ML * ml_scores +
    WEIGHT_SKILL * skill_scores +
    WEIGHT_EXP * exp_scores
)

# ================================================================
# CREATE OUTPUT
# ================================================================
final_df = pd.DataFrame({
    "Resume_ID": resume_df.index,
    "ML_Score": ml_scores.round(4),
    "Skill_Score": skill_scores.round(4),
    "Exp_Score": exp_scores.round(4),
    "Final_Score": final_scores.round(4),
})

final_df = final_df.sort_values(by="Final_Score", ascending=False)

output_path = os.path.join(OUTPUT_DIR, "final_ranked_candidates.csv")
final_df.to_csv(output_path, index=False)

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("✅ Hybrid ML + Rule-based ranking completed!")
print("=" * 60)
print(f"📁 Output: {output_path}")
print(f"📊 Total candidates ranked: {len(final_df)}")
print(f"\n🏆 Top 10 candidates:")
print(final_df.head(10).to_string(index=False))
print(f"\n📈 Score statistics:")
print(f"   ML Score      — mean: {ml_scores.mean():.4f}, max: {ml_scores.max():.4f}")
print(f"   Skill Score   — mean: {skill_scores.mean():.4f}, max: {skill_scores.max():.4f}")
print(f"   Exp Score     — mean: {exp_scores.mean():.4f}, max: {exp_scores.max():.4f}")
print(f"   Final Score   — mean: {final_scores.mean():.4f}, max: {final_scores.max():.4f}")
