"""
EXPLAINABILITY LAYER — decision_explanation_engine.py
======================================================
Patent Claim 9 + Lines 315-320:
Master engine that orchestrates trace generation + explanation for ALL
candidates after the hybrid scoring run. Produces both JSON traces and
human-readable text explanations in one pass.
"""

import pandas as pd
import os
import json
import ast
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from reasoning_trace_generator import generate_reasoning_trace, save_trace
from explanation_generator import generate_explanation


# ── Skill KB (mirrors ranking_engine/skill_matcher.py) ─────────────────
SKILL_KB = {
    "programming":    ["python", "java", "javascript", "c++", "r", "scala", "go"],
    "ml_ai":          ["machine learning", "deep learning", "neural networks",
                       "nlp", "computer vision", "reinforcement learning", "tensorflow",
                       "pytorch", "keras", "scikit-learn", "xgboost"],
    "data":           ["sql", "pandas", "numpy", "spark", "hadoop", "etl",
                       "data warehousing", "tableau", "power bi"],
    "cloud":          ["aws", "azure", "gcp", "docker", "kubernetes"],
    "web":            ["react", "node.js", "django", "flask", "html", "css"],
    "soft_skills":    ["communication", "leadership", "teamwork", "problem solving"],
}

# ── Graph inference relations (mirrors knowledge_graph) ────────────────
SKILL_GRAPH = {
    "pytorch":           ["deep learning", "neural networks", "machine learning"],
    "tensorflow":        ["deep learning", "keras", "machine learning"],
    "machine learning":  ["python", "statistics", "data analysis"],
    "deep learning":     ["neural networks", "computer vision", "nlp"],
    "spark":             ["hadoop", "big data", "scala", "python"],
    "aws":               ["cloud computing", "docker", "devops"],
    "react":             ["javascript", "html", "css", "frontend"],
    "django":            ["python", "backend", "rest api"],
    "sql":               ["database", "data warehousing", "etl"],
}


def extract_skills_from_text(text: str) -> list:
    text = text.lower()
    found = []
    for cat_skills in SKILL_KB.values():
        for skill in cat_skills:
            if skill in text:
                found.append(skill)
    return list(set(found))


def infer_skills_from_graph(resume_skills: list, job_skills: list) -> list:
    """
    For each resume skill, traverse one hop in SKILL_GRAPH to see if
    any inferred neighbour matches a job requirement that wasn't explicit.
    """
    inferred = []
    resume_lower = [s.lower() for s in resume_skills]
    job_lower    = [s.lower() for s in job_skills]
    for skill in resume_lower:
        for neighbour in SKILL_GRAPH.get(skill, []):
            if neighbour in job_lower and neighbour not in resume_lower and neighbour not in inferred:
                inferred.append(neighbour)
    return inferred


def evaluate_rules(resume_skills: list, exp_years: int,
                   job_skills: list, job_exp_min: int) -> tuple:
    """
    Apply recruitment rules and return (rule_score, activated_rules).
    Patent Claim 7: adaptive weighting adjusts rule contributions.
    """
    rules_fired = []
    score = 0.0
    weight_per_rule = 1.0 / 5  # 5 rules, each worth 0.2

    # Rule 1: Mandatory skill check (any match)
    if any(s in resume_skills for s in job_skills):
        rules_fired.append("mandatory_skill_match_rule")
        score += weight_per_rule

    # Rule 2: Minimum experience rule
    if exp_years >= job_exp_min:
        rules_fired.append("min_experience_rule")
        score += weight_per_rule

    # Rule 3: Python/programming language present
    prog_skills = SKILL_KB["programming"]
    if any(s in resume_skills for s in prog_skills):
        rules_fired.append("programming_language_rule")
        score += weight_per_rule

    # Rule 4: Domain knowledge (ML/AI)
    ml_skills = SKILL_KB["ml_ai"]
    if any(s in resume_skills for s in ml_skills):
        rules_fired.append("ml_domain_knowledge_rule")
        score += weight_per_rule

    # Rule 5: Data skills present
    data_skills = SKILL_KB["data"]
    if any(s in resume_skills for s in data_skills):
        rules_fired.append("data_skills_rule")
        score += weight_per_rule

    return round(score, 4), rules_fired


def run_decision_explanation_engine(
    ranked_csv:   str = "outputs/final_ranked_candidates.csv",
    entities_csv: str = "datasets/processed/resume_entities.csv",
    jobs_csv:     str = "datasets/processed/jobs_cleaned.csv",
    raw_jobs_csv: str = "datasets/raw/job_descriptions/job_descriptions.csv",
    output_dir:   str = "outputs",
):
    """
    Main entry point. Reads ranked candidates, generates traces +
    explanations for every candidate, and saves them.
    """
    print("Loading data for Decision Explanation Engine...")

    ranked_df   = pd.read_csv(ranked_csv)
    entities_df = pd.read_csv(entities_csv)
    jobs_df     = pd.read_csv(jobs_csv)

    # Load job skills
    try:
        raw_jobs_df = pd.read_csv(raw_jobs_csv, encoding="utf-8")
    except UnicodeDecodeError:
        raw_jobs_df = pd.read_csv(raw_jobs_csv, encoding="latin-1")

    raw_skills = str(raw_jobs_df["skills"].iloc[0]) if "skills" in raw_jobs_df.columns else ""
    job_skills = extract_skills_from_text(raw_skills)

    exp_raw    = str(jobs_df["Experience"].iloc[0]).lower()
    exp_nums   = re.findall(r"\d+", exp_raw)
    job_exp_min = int(exp_nums[0]) if len(exp_nums) >= 1 else 0
    job_exp_max = int(exp_nums[1]) if len(exp_nums) >= 2 else 99

    traces_dir      = os.path.join(output_dir, "traces")
    explanations_dir= os.path.join(output_dir, "explanations")
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(explanations_dir, exist_ok=True)

    all_traces = []
    summary_rows = []

    print(f"Generating explanations for {len(ranked_df)} candidates...\n")

    for _, row in ranked_df.iterrows():
        cid = str(row["Resume_ID"])

        # Get entity data for this candidate
        if int(row["Resume_ID"]) < len(entities_df):
            ent = entities_df.iloc[int(row["Resume_ID"])]
        else:
            ent = entities_df.iloc[0]

        # Parse resume skills
        try:
            resume_skills = ast.literal_eval(str(ent.get("skills_extracted", "[]")))
        except Exception:
            resume_skills = []

        # Parse experience
        exp_raw_cand = str(ent.get("experience", "0"))
        exp_nums_cand = re.findall(r"\d+", exp_raw_cand)
        exp_years = int(exp_nums_cand[0]) if exp_nums_cand else 0

        # Infer via graph
        inferred_skills = infer_skills_from_graph(resume_skills, job_skills)
        missing_skills  = [s for s in job_skills if s not in resume_skills and s not in inferred_skills]

        # Rule evaluation (with adaptive-style per-rule weighting)
        rule_score, activated_rules = evaluate_rules(
            resume_skills, exp_years, job_skills, job_exp_min
        )

        # Graph score = ratio of job skills covered by direct + inferred
        covered = len([s for s in job_skills if s in resume_skills or s in inferred_skills])
        graph_score = round(covered / max(len(job_skills), 1), 4)

        ml_score    = round(float(row.get("ML_Score", 0.0)), 4)
        final_score = round(
            0.50 * ml_score + 0.30 * rule_score + 0.20 * graph_score, 4
        )

        trace = generate_reasoning_trace(
            candidate_id    = cid,
            resume_skills   = resume_skills,
            job_skills      = job_skills,
            inferred_skills = inferred_skills,
            ml_score        = ml_score,
            graph_score     = graph_score,
            rule_score      = rule_score,
            final_score     = final_score,
            exp_years       = exp_years,
            job_exp_min     = job_exp_min,
            job_exp_max     = job_exp_max,
            activated_rules = activated_rules,
            missing_skills  = missing_skills,
        )

        # Save JSON trace
        trace_path = os.path.join(traces_dir, f"trace_{cid}.json")
        with open(trace_path, "w") as f:
            json.dump(trace, f, indent=2)

        # Save text explanation
        explanation = generate_explanation(trace)
        exp_path = os.path.join(explanations_dir, f"explanation_{cid}.txt")
        with open(exp_path, "w") as f:
            f.write(explanation)

        all_traces.append(trace)
        summary_rows.append({
            "Candidate_ID":      cid,
            "Final_Score":       final_score,
            "ML_Score":          ml_score,
            "Graph_Score":       graph_score,
            "Rule_Score":        rule_score,
            "Tier":              trace["tier_classification"],
            "Matched_Skills":    len(trace["skill_analysis"]["matched_skills"]),
            "Inferred_Skills":   len(inferred_skills),
            "Missing_Skills":    len(missing_skills),
            "Rules_Fired":       len(activated_rules),
            "Skill_Match_Rate":  trace["skill_analysis"]["skill_match_rate"],
        })

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows).sort_values("Final_Score", ascending=False)
    summary_path = os.path.join(output_dir, "explainable_ranked_candidates.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Traces saved      : {traces_dir}/")
    print(f"Explanations saved: {explanations_dir}/")
    print(f"Summary CSV saved : {summary_path}")
    print(f"\nTop 5 candidates:")
    print(summary_df.head(5).to_string(index=False))
    print(f"\nTOP candidates    : {(summary_df['Tier']=='TOP_CANDIDATE').sum()}")
    print(f"STANDARD candidates: {(summary_df['Tier']=='STANDARD_CANDIDATE').sum()}")

    return summary_df, all_traces


if __name__ == "__main__":
    run_decision_explanation_engine()
