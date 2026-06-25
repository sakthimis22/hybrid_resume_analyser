"""
EXPLAINABILITY LAYER — explanation_generator.py
================================================
Patent Claim 9 + Detailed Description (Lines 440-444):
Generates human-readable text explanations from a reasoning trace.
Recruiters see plain English — not just numbers.
"""

from reasoning_trace_generator import generate_reasoning_trace


def generate_explanation(trace: dict) -> str:
    """
    Convert a reasoning trace dict into a human-readable paragraph
    explanation suitable for display in the recruiter UI.
    """
    cid    = trace["candidate_id"]
    scores = trace["scores"]
    skill  = trace["skill_analysis"]
    exp    = trace["experience_analysis"]
    rules  = trace["rule_analysis"]
    graph  = trace["graph_inference"]
    tier   = trace["tier_classification"]
    contribs = trace["score_contributions"]

    lines = []

    # ── Header ──────────────────────────────────────────────────────────
    lines.append(f"=== CANDIDATE EVALUATION EXPLANATION — {cid} ===\n")
    lines.append(f"TIER: {tier.replace('_', ' ')}  |  Final Score: {scores['final_score']}\n")

    # ── Score breakdown ─────────────────────────────────────────────────
    lines.append("── SCORE BREAKDOWN ──")
    lines.append(f"  ML Similarity Score : {scores['ml_score']}  (50% weight → contributes {contribs['ml_engine_contribution (50%)']})")
    lines.append(f"  Rule-Based Score    : {scores['rule_score']}  (30% weight → contributes {contribs['rule_engine_contribution (30%)']})")
    lines.append(f"  Graph Semantic Score: {scores['graph_score']}  (20% weight → contributes {contribs['graph_engine_contribution (20%)']})")
    lines.append(f"  Formula Applied     : {contribs['fusion_formula']}\n")

    # ── Skill evidence ───────────────────────────────────────────────────
    lines.append("── SKILL ANALYSIS ──")
    if skill["matched_skills"]:
        lines.append(f"  Matched Skills ({len(skill['matched_skills'])}): {', '.join(skill['matched_skills'])}")
    else:
        lines.append("  No direct skill matches found.")

    if skill["missing_skills"]:
        lines.append(f"  Missing Skills ({len(skill['missing_skills'])}): {', '.join(skill['missing_skills'])}")

    lines.append(f"  Skill Match Rate: {round(skill['skill_match_rate']*100, 1)}%\n")

    # ── Graph inference ──────────────────────────────────────────────────
    lines.append("── INFERRED COMPETENCIES (via Knowledge Graph) ──")
    if graph["inferred_competencies"]:
        lines.append(f"  Inferred: {', '.join(graph['inferred_competencies'])}")
        lines.append(f"  Note: {graph['inference_note']}\n")
    else:
        lines.append("  No additional competencies inferred via graph.\n")

    # ── Experience ───────────────────────────────────────────────────────
    lines.append("── EXPERIENCE ANALYSIS ──")
    lines.append(f"  Status : {exp['experience_status'].replace('_', ' ')}")
    lines.append(f"  Detail : {exp['experience_note']}\n")

    # ── Rules fired ──────────────────────────────────────────────────────
    lines.append("── ACTIVATED EVALUATION RULES ──")
    if rules["activated_rules"]:
        for r in rules["activated_rules"]:
            lines.append(f"  ✔ {r}")
    else:
        lines.append("  No recruitment rules activated.")
    lines.append("")

    # ── Final verdict ────────────────────────────────────────────────────
    if tier == "TOP_CANDIDATE":
        lines.append("VERDICT: This candidate is flagged as a TOP candidate (score ≥ 0.5) "
                     "and is recommended for further recruitment stages.")
    else:
        lines.append("VERDICT: This candidate is classified as STANDARD (score < 0.5). "
                     "Consider reviewing against other applicants before proceeding.")

    return "\n".join(lines)


def generate_bulk_explanations(traces: list, output_dir: str = "outputs/explanations") -> None:
    """Generate and save explanations for a list of candidate traces."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    for trace in traces:
        explanation = generate_explanation(trace)
        path = os.path.join(output_dir, f"explanation_{trace['candidate_id']}.txt")
        with open(path, "w") as f:
            f.write(explanation)
        print(f"Explanation saved: {path}")


if __name__ == "__main__":
    # Demo
    from reasoning_trace_generator import generate_reasoning_trace
    trace = generate_reasoning_trace(
        candidate_id="CAND_001",
        resume_skills=["python", "machine learning", "sql", "pandas"],
        job_skills=["python", "machine learning", "deep learning", "sql", "tensorflow"],
        inferred_skills=["deep learning", "neural networks"],
        ml_score=0.72, graph_score=0.65, rule_score=0.80, final_score=0.737,
        exp_years=4, job_exp_min=3, job_exp_max=7,
        activated_rules=["mandatory_python_check", "min_experience_rule", "ml_domain_rule"],
        missing_skills=["tensorflow"],
    )
    print(generate_explanation(trace))
