"""
skill_inference_engine.py

Infers higher-level skill categories from lower-level resume skills
using the hierarchical skill taxonomy.

Example reasoning chain:
  tensorflow → Machine Learning → Artificial Intelligence

This gives the system knowledge-driven reasoning capability,
expanding a candidate's skill profile beyond what's explicitly listed.
"""

import json
import os

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAXONOMY_FILE = os.path.join(SCRIPT_DIR, "skill_taxonomy.json")

# Load knowledge base
with open(TAXONOMY_FILE, "r") as f:
    skill_kb = json.load(f)

# Build parent-child inference chain
# Some categories are children of broader categories
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


def infer_skills(resume_skills):
    """
    Expand resume skills by inferring parent categories from the taxonomy.

    Args:
        resume_skills (list): List of skill strings from a resume.

    Returns:
        dict: Contains original skills, inferred categories,
              full inference chain, and expanded skill set.
    """
    resume_skills_lower = [s.lower().strip() for s in resume_skills]
    inferred_categories = set()
    inference_chains = []

    # Step 1: Match skills to their direct categories
    for category, skills in skill_kb.items():
        for skill in skills:
            if skill.lower() in resume_skills_lower:
                inferred_categories.add(category)
                inference_chains.append(f"{skill} → {category}")

    # Step 2: Walk up the category hierarchy
    expanded_categories = set(inferred_categories)
    for cat in inferred_categories:
        current = cat
        while current in CATEGORY_HIERARCHY:
            parent = CATEGORY_HIERARCHY[current]
            expanded_categories.add(parent)
            inference_chains.append(f"{current} → {parent}")
            current = parent

    # Build the full expanded skill set
    expanded_skills = set(resume_skills_lower)
    for cat in expanded_categories:
        expanded_skills.add(cat.lower())

    return {
        "original_skills": resume_skills_lower,
        "direct_categories": sorted(inferred_categories),
        "all_inferred_categories": sorted(expanded_categories),
        "inference_chains": inference_chains,
        "expanded_skills": sorted(expanded_skills),
    }


# ==================== MAIN ====================
if __name__ == "__main__":

    print("=" * 60)
    print("SKILL INFERENCE ENGINE")
    print("=" * 60)

    # Test Case 1: AI/ML candidate
    print("\n--- Test 1: AI/ML Candidate ---")
    skills_1 = ["tensorflow", "python", "pandas"]
    result_1 = infer_skills(skills_1)

    print(f"Original Skills:  {result_1['original_skills']}")
    print(f"Direct Categories: {result_1['direct_categories']}")
    print(f"All Inferred:      {result_1['all_inferred_categories']}")
    print(f"Inference Chains:")
    for chain in result_1["inference_chains"]:
        print(f"   {chain}")

    # Test Case 2: Full-stack developer
    print("\n--- Test 2: Full-Stack Developer ---")
    skills_2 = ["react", "node.js", "docker", "sql", "aws"]
    result_2 = infer_skills(skills_2)

    print(f"Original Skills:  {result_2['original_skills']}")
    print(f"Direct Categories: {result_2['direct_categories']}")
    print(f"All Inferred:      {result_2['all_inferred_categories']}")
    print(f"Inference Chains:")
    for chain in result_2["inference_chains"]:
        print(f"   {chain}")

    # Test Case 3: Cybersecurity specialist
    print("\n--- Test 3: Cybersecurity Specialist ---")
    skills_3 = ["penetration testing", "firewalls", "linux", "python"]
    result_3 = infer_skills(skills_3)

    print(f"Original Skills:  {result_3['original_skills']}")
    print(f"Direct Categories: {result_3['direct_categories']}")
    print(f"All Inferred:      {result_3['all_inferred_categories']}")

    print(f"\n{'=' * 60}")
    print("Skill Inference Engine loaded successfully!")
    print(f"{'=' * 60}")
