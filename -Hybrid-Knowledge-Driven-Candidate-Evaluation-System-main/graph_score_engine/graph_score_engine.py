"""
GRAPH REASONING ENGINE — graph_score_engine.py
===============================================
Patent Claim 6: The graph reasoning engine identifies semantically
related competencies through traversal of relationships between skills,
technologies, certifications, and professional domains.

This produces an INDEPENDENT graph-based suitability score (Score 2)
that flows separately into the hybrid fusion layer — fixing GAP 3.

The skill graph is a directed adjacency structure where edges represent
"related to" or "implies knowledge of" relationships.
"""

import json
import os
from collections import deque


# ── Full skill knowledge graph ──────────────────────────────────────────
# Each key skill maps to a list of related/implied skills (one hop)
SKILL_GRAPH = {
    # ML / AI cluster
    "pytorch":            ["deep learning", "neural networks", "machine learning", "python"],
    "tensorflow":         ["deep learning", "keras", "machine learning", "python"],
    "keras":              ["deep learning", "tensorflow", "neural networks"],
    "scikit-learn":       ["machine learning", "python", "statistics"],
    "machine learning":   ["python", "statistics", "data analysis", "scikit-learn"],
    "deep learning":      ["neural networks", "computer vision", "nlp", "pytorch", "tensorflow"],
    "nlp":                ["text processing", "transformers", "python", "deep learning"],
    "computer vision":    ["image processing", "deep learning", "opencv"],
    "xgboost":            ["machine learning", "gradient boosting", "python"],
    "reinforcement learning": ["machine learning", "python", "deep learning"],

    # Data cluster
    "spark":              ["hadoop", "big data", "scala", "python", "distributed computing"],
    "hadoop":             ["big data", "hive", "mapreduce", "distributed computing"],
    "sql":                ["database", "data warehousing", "etl", "postgresql"],
    "pandas":             ["python", "data analysis", "numpy"],
    "numpy":              ["python", "scientific computing", "pandas"],
    "tableau":            ["data visualization", "business intelligence", "sql"],
    "power bi":           ["data visualization", "business intelligence", "dax"],

    # Cloud / DevOps cluster
    "aws":                ["cloud computing", "docker", "devops", "s3", "ec2"],
    "azure":              ["cloud computing", "devops", "microsoft", "docker"],
    "gcp":                ["cloud computing", "bigquery", "devops", "docker"],
    "docker":             ["containerization", "kubernetes", "devops", "microservices"],
    "kubernetes":         ["docker", "devops", "microservices", "orchestration"],

    # Programming cluster
    "python":             ["programming", "scripting", "automation"],
    "java":               ["programming", "oop", "spring", "backend"],
    "javascript":         ["programming", "frontend", "node.js", "react"],
    "react":              ["javascript", "frontend", "html", "css"],
    "django":             ["python", "backend", "rest api", "web development"],
    "flask":              ["python", "backend", "rest api", "web development"],
    "node.js":            ["javascript", "backend", "rest api"],

    # Soft skills / domain
    "leadership":         ["management", "teamwork", "communication"],
    "agile":              ["scrum", "project management", "teamwork"],
    "statistics":         ["mathematics", "probability", "data analysis"],
}

DOMAIN_CERTIFICATIONS = {
    "aws certified":     ["aws", "cloud computing"],
    "google cloud":      ["gcp", "cloud computing"],
    "pmp":               ["project management", "agile"],
    "tensorflow developer": ["tensorflow", "deep learning"],
    "azure certified":   ["azure", "cloud computing"],
}


def build_graph() -> dict:
    """Return the skill graph (extendable from JSON file if present)."""
    graph_path = "knowledge_base/skill_graph.json"
    if os.path.exists(graph_path):
        with open(graph_path) as f:
            external = json.load(f)
        merged = dict(SKILL_GRAPH)
        merged.update(external)
        return merged
    return SKILL_GRAPH


def bfs_infer(seed_skills: list, graph: dict, max_hops: int = 2) -> set:
    """
    BFS traversal from seed skills up to max_hops.
    Returns all reachable skills (inferred competencies).
    """
    visited = set(s.lower() for s in seed_skills)
    queue   = deque((s.lower(), 0) for s in seed_skills)
    inferred = set()

    while queue:
        skill, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbour in graph.get(skill, []):
            n = neighbour.lower()
            if n not in visited:
                visited.add(n)
                inferred.add(n)
                queue.append((n, depth + 1))

    return inferred


def compute_graph_score(
    resume_skills: list,
    job_skills:    list,
    max_hops:      int = 2,
) -> tuple:
    """
    Compute the independent graph-based suitability score (Score 2).

    Returns:
        graph_score     (float 0–1)
        inferred_skills (list) — competencies found via graph but not in resume
        coverage_detail (dict) — breakdown for explainability
    """
    graph = build_graph()
    resume_lower = set(s.lower() for s in resume_skills)
    job_lower    = set(s.lower() for s in job_skills)

    # Direct matches
    direct_matches = resume_lower & job_lower

    # Inferred via BFS
    all_reachable = bfs_infer(list(resume_lower), graph, max_hops)
    inferred_matches = (all_reachable & job_lower) - direct_matches
    inferred_skills  = sorted(inferred_matches)

    # Skills still missing even after inference
    missing = job_lower - direct_matches - inferred_matches

    # Score = fraction of job skills covered (direct + inferred)
    covered     = len(direct_matches) + len(inferred_matches)
    total_needed= max(len(job_lower), 1)
    # Direct match worth full credit; inferred worth 0.7 credit
    weighted_covered = len(direct_matches) + 0.7 * len(inferred_matches)
    graph_score = round(min(weighted_covered / total_needed, 1.0), 4)

    coverage_detail = {
        "job_skills":          sorted(job_lower),
        "resume_skills":       sorted(resume_lower),
        "direct_matches":      sorted(direct_matches),
        "inferred_matches":    inferred_skills,
        "missing_skills":      sorted(missing),
        "coverage_fraction":   f"{covered}/{total_needed}",
        "graph_score":         graph_score,
        "max_hops_used":       max_hops,
    }

    return graph_score, inferred_skills, coverage_detail


def batch_graph_scores(candidates: list, job_skills: list) -> list:
    """
    Compute graph scores for a list of candidates.

    candidates: list of dicts with keys "candidate_id" and "resume_skills"
    Returns list of result dicts.
    """
    results = []
    for cand in candidates:
        score, inferred, detail = compute_graph_score(
            cand["resume_skills"], job_skills
        )
        results.append({
            "candidate_id":     cand["candidate_id"],
            "graph_score":      score,
            "inferred_skills":  inferred,
            "coverage_detail":  detail,
        })
    return results


if __name__ == "__main__":
    # Demo
    job_skills     = ["python", "machine learning", "tensorflow", "deep learning", "sql", "aws"]
    resume_skills  = ["python", "pytorch", "pandas", "sql"]

    score, inferred, detail = compute_graph_score(resume_skills, job_skills)

    print(f"Graph Score  : {score}")
    print(f"Inferred via graph: {inferred}")
    print(f"\nCoverage detail:")
    for k, v in detail.items():
        print(f"  {k}: {v}")
