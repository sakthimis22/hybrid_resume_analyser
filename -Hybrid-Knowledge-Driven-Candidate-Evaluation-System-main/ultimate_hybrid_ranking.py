"""
ultimate_hybrid_ranking.py
─────────────────────────────────────────────────────────────────
Patent Layer 4 — Hybrid Decision Fusion Layer (COMPLETE VERSION)
Combines:
  Score 1 → ML Engine     (TF-IDF + Cosine Similarity)   50%
  Score 2 → Graph Engine  (Skill Graph Reasoning)         20%
  Score 3 → Rule Engine   (Adaptive Rule Weighting)       30%
  Bonus   → Experience alignment                         +10%
  Bonus   → Project relevance                            +10%

Also implements:
  ✅ TOP vs STANDARD candidate classification (score ≥ 0.50)
  ✅ Proper 3-engine fusion (not merged skill/exp)
  ✅ Adaptive rule weights loaded dynamically
  ✅ Outputs final_ranked_candidates.csv with all score columns

Run:
    python ultimate_hybrid_ranking.py
─────────────────────────────────────────────────────────────────
"""

import os
import re
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Import project modules ────────────────────────────────────────
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from knowledge_graph.graph_skill_reasoning import compute_graph_score
from reasoning_engine.adaptive_rule_weighting import (
    load_weights, compute_rule_score
)

# ── Paths ─────────────────────────────────────────────────────────
PROCESSED_DIR = "datasets/processed"
RESUME_FILE   = os.path.join(PROCESSED_DIR, "resumes_cleaned.csv")
ENTITIES_FILE = os.path.join(PROCESSED_DIR, "resume_entities.csv")
RAW_JOB_FILE  = "datasets/raw/job_descriptions/job_descriptions.csv"
JOB_FILE      = os.path.join(PROCESSED_DIR, "jobs_cleaned.csv")
OUTPUT_DIR    = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Fusion weights (Patent Figure 2 specification) ─────────────────
W_ML    = 0.50
W_GRAPH = 0.20
W_RULE  = 0.30
BONUS_EXP  = 0.10   # additive bonus
BONUS_PROJ = 0.10   # additive bonus

# ── TOP candidate threshold (Patent Figure 1) ─────────────────────
TOP_THRESHOLD = 0.50

# ── Skill Knowledge Base (for rule evaluation) ────────────────────
SKILL_KB = {
    "programming": ["python","java","c++","javascript","r","scala","golang"],
    "ml_ai":       ["machine learning","deep learning","nlp","computer vision",
                    "tensorflow","pytorch","scikit-learn","keras","transformers",
                    "data science"],
    "data":        ["sql","nosql","mongodb","postgresql","mysql","spark",
                    "hadoop","etl","data engineering","data warehouse"],
    "cloud":       ["aws","gcp","azure","docker","kubernetes","terraform",
                    "devops","ci/cd","mlops"],
}

RULES = {
    "min_experience_met":    lambda e, mn, mx: e >= mn,
    "experience_not_over":   lambda e, mn, mx: e <= mx + 3,
    "has_programming_skill": lambda s, *_: any(x in s for x in SKILL_KB["programming"]),
    "has_ml_skill":          lambda s, *_: any(x in s for x in SKILL_KB["ml_ai"]),
    "has_data_skill":        lambda s, *_: any(x in s for x in SKILL_KB["data"]),
    "has_cloud_skill":       lambda s, *_: any(x in s for x in SKILL_KB["cloud"]),
}


# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def experience_score(candidate_yrs, job_min, job_max) -> float:
    if job_min <= candidate_yrs <= job_max:
        return 1.0
    elif candidate_yrs < job_min:
        return max(0.0, candidate_yrs / job_min) if job_min > 0 else 0.0
    else:
        return max(0.5, 1.0 - (candidate_yrs - job_max) / 20)


def project_score(entities_row) -> float:
    """Simple heuristic: presence of project info gives bonus."""
    proj = str(entities_row.get("projects", "")).strip()
    if proj and proj.lower() not in ("nan", "none", "[]", ""):
        return 1.0
    return 0.0


def evaluate_rules(resume_skills, exp_years, job_min, job_max, weights) -> tuple:
    fired, failed = [], []
    for name, fn in RULES.items():
        try:
            if "experience" in name:
                ok = fn(exp_years, job_min, job_max)
            else:
                ok = fn(resume_skills)
            (fired if ok else failed).append(name)
        except Exception:
            failed.append(name)
    rule_sc = compute_rule_score(fired, weights)
    return rule_sc, fired, failed


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────

def run(job_skills=None, job_exp_min=2, job_exp_max=8):
    print("=" * 60)
    print("  HYBRID DECISION FUSION ENGINE")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────
    resume_df   = pd.read_csv(RESUME_FILE)
    entities_df = pd.read_csv(ENTITIES_FILE) if os.path.exists(ENTITIES_FILE) else None
    job_df      = pd.read_csv(JOB_FILE)

    try:
        raw_job_df = pd.read_csv(RAW_JOB_FILE, encoding="utf-8")
    except Exception:
        raw_job_df = pd.read_csv(RAW_JOB_FILE, encoding="latin-1")

    # Build job text
    raw_skills = str(raw_job_df["skills"].iloc[0]) if "skills" in raw_job_df.columns else ""
    raw_resp   = str(raw_job_df["Responsibilities"].iloc[0]) if "Responsibilities" in raw_job_df.columns else ""
    job_text   = clean_text(raw_skills + " " + raw_resp)

    if job_skills is None:
        job_skills = [w for w in job_text.split() if len(w) > 3][:30]

    # Parse experience range from job file
    exp_raw  = str(job_df["Experience"].iloc[0]).lower()
    exp_nums = re.findall(r"\d+", exp_raw)
    if len(exp_nums) >= 2:
        job_exp_min, job_exp_max = int(exp_nums[0]), int(exp_nums[1])
    elif len(exp_nums) == 1:
        job_exp_min = job_exp_max = int(exp_nums[0])

    print(f"\n  Job experience required : {job_exp_min}–{job_exp_max} years")
    print(f"  Job skills (sample)     : {job_skills[:8]}")

    # ── 2. ML Score (Score 1) ─────────────────────────────────────
    print("\n[1/4] Computing ML similarity scores...")
    resume_texts = resume_df["Resume_str"].apply(clean_text) if "Resume_str" in resume_df.columns \
                   else resume_df.iloc[:, 0].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    combined = pd.concat([resume_texts, pd.Series([job_text])], ignore_index=True)
    tfidf.fit(combined)
    r_vecs  = tfidf.transform(resume_texts)
    j_vec   = tfidf.transform([job_text])
    ml_scores = cosine_similarity(r_vecs, j_vec).flatten()

    # ── 3. Graph Score (Score 2) ──────────────────────────────────
    print("[2/4] Computing graph-based suitability scores...")
    graph_scores  = []
    inferred_list = []
    for i, row in resume_df.iterrows():
        resume_skills = []
        if entities_df is not None and i < len(entities_df):
            try:
                resume_skills = ast.literal_eval(
                    str(entities_df.iloc[i].get("skills_extracted", "[]"))
                )
            except Exception:
                resume_skills = []
        gs, inferred = compute_graph_score(resume_skills, job_skills)
        graph_scores.append(gs)
        inferred_list.append(inferred)

    # ── 4. Rule Score (Score 3) with adaptive weights ─────────────
    print("[3/4] Computing adaptive rule-based scores...")
    adaptive_weights = load_weights()
    rule_scores = []
    rule_fired_list, rule_failed_list = [], []
    exp_scores, proj_scores = [], []

    for i, row in resume_df.iterrows():
        resume_skills, exp_years = [], 0
        proj_info = {}
        if entities_df is not None and i < len(entities_df):
            ent = entities_df.iloc[i]
            try:
                resume_skills = ast.literal_eval(str(ent.get("skills_extracted", "[]")))
            except Exception:
                resume_skills = []
            exp_nums2 = re.findall(r"\d+", str(ent.get("experience", "0")))
            exp_years = int(exp_nums2[0]) if exp_nums2 else 0
            proj_info = ent

        rs, fired, failed = evaluate_rules(
            resume_skills, exp_years, job_exp_min, job_exp_max, adaptive_weights
        )
        rule_scores.append(rs)
        rule_fired_list.append(fired)
        rule_failed_list.append(failed)
        exp_scores.append(experience_score(exp_years, job_exp_min, job_exp_max))
        proj_scores.append(project_score(proj_info) if entities_df is not None else 0.5)

    # ── 5. Hybrid Fusion ──────────────────────────────────────────
    print("[4/4] Applying hybrid decision fusion formula...")
    print(f"      Formula: {W_ML}×ML + {W_GRAPH}×Graph + {W_RULE}×Rule "
          f"+ {BONUS_EXP}×Exp + {BONUS_PROJ}×Project")

    import numpy as np
    ml_arr    = ml_scores
    graph_arr = np.array(graph_scores)
    rule_arr  = np.array(rule_scores)
    exp_arr   = np.array(exp_scores)
    proj_arr  = np.array(proj_scores)

    final_scores = (
        W_ML    * ml_arr    +
        W_GRAPH * graph_arr +
        W_RULE  * rule_arr  +
        BONUS_EXP  * exp_arr  +
        BONUS_PROJ * proj_arr
    )
    # Clip to [0, 1]
    final_scores = np.clip(final_scores, 0, 1)

    # ── 6. Classification — TOP vs STANDARD ──────────────────────
    labels = ["TOP" if s >= TOP_THRESHOLD else "STANDARD" for s in final_scores]

    # ── 7. Build output dataframe ─────────────────────────────────
    out_df = pd.DataFrame({
        "Resume_ID":    resume_df.index,
        "ML_Score":     ml_arr.round(4),
        "Graph_Score":  graph_arr.round(4),
        "Rule_Score":   rule_arr.round(4),
        "Exp_Score":    exp_arr.round(4),
        "Proj_Score":   proj_arr.round(4),
        "Final_Score":  final_scores.round(4),
        "Label":        labels,
        "Rules_Fired":  ["|".join(r) for r in rule_fired_list],
        "Rules_Failed": ["|".join(r) for r in rule_failed_list],
        "Inferred_Skills": [",".join(inf) for inf in inferred_list],
    })
    out_df = out_df.sort_values("Final_Score", ascending=False).reset_index(drop=True)
    out_df.insert(0, "Rank", out_df.index + 1)

    # ── 8. Save ───────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "final_ranked_candidates.csv")
    out_df.to_csv(out_path, index=False)

    top_count = (out_df["Label"] == "TOP").sum()
    print(f"\n{'='*60}")
    print(f"✅ Hybrid fusion complete!")
    print(f"   Total candidates ranked : {len(out_df)}")
    print(f"   TOP candidates          : {top_count}  (score ≥ {TOP_THRESHOLD})")
    print(f"   STANDARD candidates     : {len(out_df) - top_count}")
    print(f"   Output saved            : {out_path}")
    print(f"\n🏆 Top 10:")
    print(out_df[["Rank","Resume_ID","ML_Score","Graph_Score",
                  "Rule_Score","Final_Score","Label"]].head(10).to_string(index=False))
    return out_df


if __name__ == "__main__":
    run(
        job_skills=["python", "machine learning", "sql",
                    "docker", "aws", "data science"],
        job_exp_min=2,
        job_exp_max=8
    )
