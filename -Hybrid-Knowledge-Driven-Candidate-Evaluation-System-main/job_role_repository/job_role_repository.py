"""
JOB ROLE REPOSITORY — job_role_repository.py
=============================================
Patent Lines 221-225: The system retrieves predefined skill requirements
and competency profiles corresponding to the selected role from a
job-role repository.

This module provides:
1. A built-in job_roles.json-equivalent structure
2. Role lookup by name (exact and fuzzy)
3. Integration point for the UI layer (recruiter selects a role)
4. Extendable — new roles can be added to the JSON file
"""

import json
import os


# ── Built-in role competency profiles ──────────────────────────────────
# Mirrors what job_roles.json should contain
BUILTIN_ROLES = {
    "data_scientist": {
        "title": "Data Scientist",
        "required_skills": [
            "python", "machine learning", "statistics", "sql",
            "pandas", "numpy", "scikit-learn", "data visualization"
        ],
        "preferred_skills": ["deep learning", "tensorflow", "pytorch", "spark", "r"],
        "required_certifications": [],
        "min_experience_years": 2,
        "max_experience_years": 8,
        "education": "Bachelor's or Master's in CS, Statistics, or related field",
        "domain": "data_science",
        "mandatory_skills": ["python", "machine learning"],
    },
    "ml_engineer": {
        "title": "Machine Learning Engineer",
        "required_skills": [
            "python", "machine learning", "deep learning", "tensorflow",
            "pytorch", "docker", "sql", "git"
        ],
        "preferred_skills": ["kubernetes", "aws", "spark", "mlops", "airflow"],
        "required_certifications": [],
        "min_experience_years": 3,
        "max_experience_years": 10,
        "education": "Bachelor's in CS or Engineering",
        "domain": "machine_learning",
        "mandatory_skills": ["python", "machine learning", "deep learning"],
    },
    "data_engineer": {
        "title": "Data Engineer",
        "required_skills": [
            "python", "sql", "spark", "hadoop", "etl",
            "data warehousing", "airflow", "docker"
        ],
        "preferred_skills": ["aws", "azure", "kafka", "scala", "kubernetes"],
        "required_certifications": [],
        "min_experience_years": 2,
        "max_experience_years": 8,
        "education": "Bachelor's in CS or Engineering",
        "domain": "data_engineering",
        "mandatory_skills": ["python", "sql", "spark"],
    },
    "software_engineer": {
        "title": "Software Engineer",
        "required_skills": [
            "python", "java", "javascript", "sql", "git",
            "rest api", "oop", "data structures", "algorithms"
        ],
        "preferred_skills": ["docker", "kubernetes", "aws", "react", "microservices"],
        "required_certifications": [],
        "min_experience_years": 1,
        "max_experience_years": 8,
        "education": "Bachelor's in CS or related field",
        "domain": "software_engineering",
        "mandatory_skills": ["python", "git"],
    },
    "nlp_engineer": {
        "title": "NLP Engineer",
        "required_skills": [
            "python", "nlp", "transformers", "deep learning",
            "pytorch", "text processing", "machine learning"
        ],
        "preferred_skills": ["huggingface", "spacy", "nltk", "tensorflow", "bert"],
        "required_certifications": [],
        "min_experience_years": 2,
        "max_experience_years": 7,
        "education": "Master's or PhD in CS, Linguistics, or AI",
        "domain": "nlp",
        "mandatory_skills": ["python", "nlp", "transformers"],
    },
    "cloud_architect": {
        "title": "Cloud Architect",
        "required_skills": [
            "aws", "azure", "gcp", "docker", "kubernetes",
            "terraform", "networking", "security", "devops"
        ],
        "preferred_skills": ["python", "ansible", "ci_cd", "microservices"],
        "required_certifications": ["aws certified", "azure certified", "google cloud"],
        "min_experience_years": 5,
        "max_experience_years": 15,
        "education": "Bachelor's in CS or Engineering",
        "domain": "cloud",
        "mandatory_skills": ["aws", "docker", "kubernetes"],
    },
    "business_analyst": {
        "title": "Business Analyst",
        "required_skills": [
            "sql", "tableau", "power bi", "excel",
            "communication", "requirements gathering", "agile"
        ],
        "preferred_skills": ["python", "data analysis", "jira", "scrum"],
        "required_certifications": [],
        "min_experience_years": 1,
        "max_experience_years": 7,
        "education": "Bachelor's in Business, CS, or related field",
        "domain": "business_analysis",
        "mandatory_skills": ["sql", "communication"],
    },
    "devops_engineer": {
        "title": "DevOps Engineer",
        "required_skills": [
            "docker", "kubernetes", "ci_cd", "git", "linux",
            "ansible", "terraform", "aws", "python", "bash"
        ],
        "preferred_skills": ["jenkins", "gitlab_ci", "prometheus", "grafana"],
        "required_certifications": [],
        "min_experience_years": 2,
        "max_experience_years": 8,
        "education": "Bachelor's in CS or Engineering",
        "domain": "devops",
        "mandatory_skills": ["docker", "kubernetes", "ci_cd"],
    },
}

ROLES_FILE = "knowledge_base/job_roles.json"


def load_roles() -> dict:
    """Load roles from JSON file if present, else use built-in profiles."""
    if os.path.exists(ROLES_FILE):
        with open(ROLES_FILE) as f:
            external = json.load(f)
        merged = dict(BUILTIN_ROLES)
        merged.update(external)
        return merged
    return BUILTIN_ROLES.copy()


def list_available_roles() -> list:
    """Return list of available role keys for UI dropdown."""
    roles = load_roles()
    return [{"key": k, "title": v["title"], "domain": v["domain"]}
            for k, v in roles.items()]


def get_role_profile(role_key: str) -> dict:
    """
    Retrieve competency profile for a named role.
    Patent Lines 221-225: system retrieves predefined skill requirements.
    """
    roles = load_roles()
    role_key = role_key.lower().replace(" ", "_")

    if role_key in roles:
        return roles[role_key]

    # Fuzzy match — find closest role name
    for key in roles:
        if role_key in key or key in role_key:
            print(f"  Fuzzy matched '{role_key}' to '{key}'")
            return roles[key]

    raise KeyError(f"Role '{role_key}' not found. Available: {list(roles.keys())}")


def get_job_skills_for_role(role_key: str) -> list:
    """Convenience: return combined required + preferred skills for a role."""
    profile = get_role_profile(role_key)
    return list(set(profile["required_skills"] + profile.get("preferred_skills", [])))


def save_custom_role(role_key: str, profile: dict) -> None:
    """Allow recruiters to add custom role profiles that persist to disk."""
    os.makedirs(os.path.dirname(ROLES_FILE), exist_ok=True)

    existing = {}
    if os.path.exists(ROLES_FILE):
        with open(ROLES_FILE) as f:
            existing = json.load(f)

    existing[role_key.lower().replace(" ", "_")] = profile

    with open(ROLES_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Custom role '{role_key}' saved to {ROLES_FILE}")


def print_role_summary(role_key: str) -> None:
    profile = get_role_profile(role_key)
    print(f"\n=== ROLE PROFILE: {profile['title']} ===")
    print(f"  Domain              : {profile['domain']}")
    print(f"  Experience Required : {profile['min_experience_years']}–{profile['max_experience_years']} years")
    print(f"  Education           : {profile['education']}")
    print(f"  Required Skills     : {', '.join(profile['required_skills'])}")
    print(f"  Preferred Skills    : {', '.join(profile.get('preferred_skills', []))}")
    print(f"  Mandatory Skills    : {', '.join(profile.get('mandatory_skills', []))}")
    if profile.get("required_certifications"):
        print(f"  Certifications      : {', '.join(profile['required_certifications'])}")


if __name__ == "__main__":
    print("Available roles:")
    for r in list_available_roles():
        print(f"  [{r['key']}] {r['title']} ({r['domain']})")

    print_role_summary("data_scientist")
    print_role_summary("ml_engineer")

    # Demo custom role add
    save_custom_role("prompt_engineer", {
        "title": "Prompt Engineer",
        "required_skills": ["python", "nlp", "llm", "prompt design", "transformers"],
        "preferred_skills": ["langchain", "openai api", "rag", "fine tuning"],
        "required_certifications": [],
        "min_experience_years": 1,
        "max_experience_years": 5,
        "education": "Bachelor's in CS or related field",
        "domain": "generative_ai",
        "mandatory_skills": ["python", "llm"],
    })
    print("\nCustom role added successfully.")
