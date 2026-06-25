"""
skill_matcher.py

Matches resume skills against job skills using the skill inference engine.
Inferred skills (e.g., tensorflow → Machine Learning) are included in
the comparison, making the matching knowledge-driven.
"""

import json
import os

# Resolve paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TAXONOMY_FILE = os.path.join(PROJECT_ROOT, "knowledge_base", "skill_taxonomy.json")

# Load knowledge base
with open(TAXONOMY_FILE, "r") as f:
    skill_kb = json.load(f)

# Category hierarchy for multi-level inference
CATEGORY_HIERARCHY = {
    "Machine Learning": "Artificial Intelligence",
    "Data Science": "Artificial Intelligence",
    "Big Data & Data Engineering": "Data Science",
    "Backend Development": "Software Engineering",
    "Web Development": "Software Engineering",
    "Mobile Development": "Software Engineering",
    "DevOps & CI/CD": "Cloud Computing",
    "Monitoring & Observability": "Cloud Computing",
    "Testing & QA": "Software Engineering",
    "Embedded Systems & IoT": "Engineering",
    "Blockchain & Web3": "Software Engineering",
    "Digital Marketing & SEO": "Business & Domain Skills",
}


def infer_skills(skills):
    """
    Expand a skill list by inferring parent categories from the taxonomy.

    Example:
        ["tensorflow", "python"] → adds "Machine Learning",
        "Data Science", "Artificial Intelligence"
    """
    skills_lower = set(s.lower().strip() for s in skills)
    inferred = set(skills_lower)

    # Map skills to direct categories
    matched_categories = set()
    for category, subskills in skill_kb.items():
        for s in subskills:
            if s.lower() in skills_lower:
                matched_categories.add(category)
                inferred.add(category.lower())

    # Walk up the category hierarchy
    for cat in list(matched_categories):
        current = cat
        while current in CATEGORY_HIERARCHY:
            parent = CATEGORY_HIERARCHY[current]
            inferred.add(parent.lower())
            current = parent

    return list(inferred)


def skill_match_score(resume_skills, job_skills):
    """
    Compute a match score between resume skills and job skills.

    Uses inference to expand resume skills before matching.
    Score = number of matched skills / total job skills required.

    Args:
        resume_skills (list): Skills extracted from a resume.
        job_skills (list): Skills required by the job.

    Returns:
        float: Match ratio between 0.0 and 1.0.
    """
    if not job_skills:
        return 0.0

    # Expand resume skills with inferred categories
    resume_all = set(s.lower().strip() for s in infer_skills(resume_skills))
    job_set = set(s.lower().strip() for s in job_skills)

    matched = resume_all & job_set

    return len(matched) / len(job_set)


# ==================== MAIN ====================
if __name__ == "__main__":

    print("=" * 60)
    print("SKILL MATCHER WITH INFERENCE")
    print("=" * 60)

    # Test 1: tensorflow should infer machine learning
    resume = ["tensorflow", "python"]
    job = ["machine learning", "python"]
    score = skill_match_score(resume, job)
    print(f"\nTest 1:")
    print(f"  Resume: {resume}")
    print(f"  Job:    {job}")
    print(f"  Score:  {score}")
    print(f"  (tensorflow infers 'machine learning' -> full match)")

    # Test 2: No inference needed
    resume2 = ["java", "sql"]
    job2 = ["python", "machine learning"]
    score2 = skill_match_score(resume2, job2)
    print(f"\nTest 2:")
    print(f"  Resume: {resume2}")
    print(f"  Job:    {job2}")
    print(f"  Score:  {score2}")

    # Test 3: Partial match with inference
    resume3 = ["react", "node.js", "docker"]
    job3 = ["web development", "cloud computing", "java", "sql"]
    score3 = skill_match_score(resume3, job3)
    print(f"\nTest 3:")
    print(f"  Resume: {resume3}")
    print(f"  Job:    {job3}")
    print(f"  Score:  {score3}")
    print(f"  (react infers 'web development', docker infers 'cloud computing')")

    print(f"\n{'=' * 60}")
    print("Skill Matcher ready!")
