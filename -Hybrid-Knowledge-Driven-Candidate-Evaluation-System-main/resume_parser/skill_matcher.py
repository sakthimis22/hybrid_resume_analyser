"""
skill_matcher.py

Utility module for matching candidate skills against job requirements.
Can be used downstream by the ranking pipeline to compute skill-based
similarity scores between resumes and job descriptions.
"""

import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_skill_list():
    """Load the master skill list from the skills dataset."""
    skills_path = os.path.join(PROJECT_ROOT, "datasets", "raw", "skills", "IT_Job_Roles_Skills.csv")
    skills = pd.read_csv(skills_path, encoding="latin-1")
    return [skill.strip().lower() for skill in skills['Skills'].dropna().tolist()]


def match_skills(candidate_skills, required_skills):
    """
    Compare candidate skills against required skills for a job.

    Args:
        candidate_skills (list): List of skills extracted from resume.
        required_skills (list): List of skills required by the job.

    Returns:
        dict: Match results with matched skills, missing skills, and match ratio.
    """
    candidate_set = set(s.lower().strip() for s in candidate_skills)
    required_set = set(s.lower().strip() for s in required_skills)

    matched = candidate_set & required_set
    missing = required_set - candidate_set

    match_ratio = len(matched) / len(required_set) if required_set else 0.0

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "match_ratio": round(match_ratio, 4),
        "matched_count": len(matched),
        "required_count": len(required_set),
    }


if __name__ == "__main__":
    # Quick demo
    candidate = ["python", "sql", "machine learning", "docker"]
    required = ["python", "sql", "aws", "machine learning", "tensorflow"]

    result = match_skills(candidate, required)
    print("Skill Match Results:")
    print(f"  Matched:  {result['matched_skills']}")
    print(f"  Missing:  {result['missing_skills']}")
    print(f"  Ratio:    {result['match_ratio']}")
