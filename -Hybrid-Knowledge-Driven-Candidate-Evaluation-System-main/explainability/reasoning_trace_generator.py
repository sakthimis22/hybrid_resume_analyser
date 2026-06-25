"""
EXPLAINABILITY LAYER — reasoning_trace_generator.py
=====================================================
Patent Claim 9: Generates a reasoning trace identifying matched skills,
inferred competencies, and activated evaluation rules associated with
the unified candidate suitability score.
"""

import json
import os
from datetime import datetime


def generate_reasoning_trace(
    candidate_id,
    resume_skills: list,
    job_skills: list,
    inferred_skills: list,
    ml_score: float,
    graph_score: float,
    rule_score: float,
    final_score: float,
    exp_years: int,
    job_exp_min: int,
    job_exp_max: int,
    activated_rules: list,
    missing_skills: list,
) -> dict:
    """
    Generate a structured reasoning trace for one candidate evaluation.
    Patent Claim 9 compliance: matched skills, inferred competencies,
    activated rules are all identified explicitly.
    """

    matched_skills = [s for s in resume_skills if s.lower() in [j.lower() for j in job_skills]]
    match_rate = round(len(matched_skills) / max(len(job_skills), 1), 4)

    if job_exp_min <= exp_years <= job_exp_max:
        exp_status = "WITHIN_RANGE"
        exp_note = f"Candidate has {exp_years} yrs — fits required {job_exp_min}–{job_exp_max} yrs perfectly."
    elif exp_years < job_exp_min:
        exp_status = "UNDER_QUALIFIED"
        exp_note = f"Candidate has {exp_years} yrs — below required minimum of {job_exp_min} yrs."
    else:
        exp_status = "OVER_QUALIFIED"
        exp_note = f"Candidate has {exp_years} yrs — exceeds required maximum of {job_exp_max} yrs."

    ml_contribution    = round(0.50 * ml_score,    4)
    rule_contribution  = round(0.30 * rule_score,  4)
    graph_contribution = round(0.20 * graph_score, 4)
    tier = "TOP_CANDIDATE" if final_score >= 0.5 else "STANDARD_CANDIDATE"

    trace = {
        "trace_id":            f"TRACE-{candidate_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "candidate_id":        candidate_id,
        "generated_at":        datetime.now().isoformat(),
        "tier_classification": tier,
        "scores": {
            "ml_score":    round(ml_score,    4),
            "graph_score": round(graph_score, 4),
            "rule_score":  round(rule_score,  4),
            "final_score": round(final_score, 4),
        },
        "score_contributions": {
            "ml_engine_contribution (50%)":    ml_contribution,
            "rule_engine_contribution (30%)":  rule_contribution,
            "graph_engine_contribution (20%)": graph_contribution,
            "fusion_formula": "Final = 0.50×ML + 0.30×Rule + 0.20×Graph",
        },
        "skill_analysis": {
            "job_required_skills":   job_skills,
            "candidate_skills":      resume_skills,
            "matched_skills":        matched_skills,
            "missing_skills":        missing_skills,
            "inferred_competencies": inferred_skills,
            "skill_match_rate":      match_rate,
        },
        "experience_analysis": {
            "candidate_experience_years": exp_years,
            "job_required_min_years":     job_exp_min,
            "job_required_max_years":     job_exp_max,
            "experience_status":          exp_status,
            "experience_note":            exp_note,
        },
        "rule_analysis": {
            "activated_rules":   activated_rules,
            "total_rules_fired": len(activated_rules),
        },
        "graph_inference": {
            "inferred_competencies": inferred_skills,
            "inference_note": (
                "These competencies were NOT explicitly in the resume but were "
                "inferred via the skill knowledge graph by traversing related skill edges."
            ) if inferred_skills else "No additional competencies inferred.",
        },
    }
    return trace


def save_trace(trace: dict, output_dir: str = "outputs/traces") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{trace['trace_id']}.json")
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)
    return path


if __name__ == "__main__":
    sample = generate_reasoning_trace(
        candidate_id="CAND_001",
        resume_skills=["python", "machine learning", "sql", "pandas"],
        job_skills=["python", "machine learning", "deep learning", "sql", "tensorflow"],
        inferred_skills=["deep learning", "neural networks"],
        ml_score=0.72, graph_score=0.65, rule_score=0.80, final_score=0.737,
        exp_years=4, job_exp_min=3, job_exp_max=7,
        activated_rules=["mandatory_python_check", "min_experience_rule", "ml_domain_rule"],
        missing_skills=["tensorflow"],
    )
    path = save_trace(sample)
    print(f"Reasoning trace saved: {path}")
    print(json.dumps(sample, indent=2))
