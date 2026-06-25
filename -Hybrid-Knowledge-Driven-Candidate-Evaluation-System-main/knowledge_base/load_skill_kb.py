"""
load_skill_kb.py

Loads the hierarchical skill taxonomy from skill_taxonomy.json.
Provides:
  - Category listing with skill counts
  - Reverse lookup: skill → parent category
  - Coverage validation against the actual skills dataset
"""

import json
import os
import pandas as pd

# Resolve paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TAXONOMY_FILE = os.path.join(SCRIPT_DIR, "skill_taxonomy.json")
SKILLS_CSV = os.path.join(PROJECT_ROOT, "datasets", "raw", "skills", "IT_Job_Roles_Skills.csv")


def load_taxonomy():
    """Load the skill taxonomy from JSON."""
    with open(TAXONOMY_FILE, "r") as f:
        return json.load(f)


def build_reverse_lookup(taxonomy):
    """Build a reverse mapping: skill → list of parent categories."""
    reverse = {}
    for category, skills in taxonomy.items():
        for skill in skills:
            skill_lower = skill.lower().strip()
            if skill_lower not in reverse:
                reverse[skill_lower] = []
            reverse[skill_lower].append(category)
    return reverse


def get_skill_categories(skill_name, reverse_lookup):
    """Look up which categories a skill belongs to."""
    return reverse_lookup.get(skill_name.lower().strip(), [])


def infer_categories(skills_list, reverse_lookup):
    """Given a list of skills, infer all related categories."""
    categories = set()
    for skill in skills_list:
        cats = get_skill_categories(skill, reverse_lookup)
        categories.update(cats)
    return sorted(categories)


# ==================== MAIN ====================
if __name__ == "__main__":
    # Load taxonomy
    skill_kb = load_taxonomy()
    reverse_lookup = build_reverse_lookup(skill_kb)

    # Print all categories
    print("=" * 60)
    print("📚 SKILL KNOWLEDGE BASE")
    print("=" * 60)

    total_skills = 0
    for category, skills in skill_kb.items():
        print(f"\n📂 {category} ({len(skills)} skills)")
        for skill in skills:
            print(f"   • {skill}")
        total_skills += len(skills)

    print(f"\n{'=' * 60}")
    print(f"📊 Total categories: {len(skill_kb)}")
    print(f"📊 Total skill entries: {total_skills}")
    print(f"📊 Unique skills in reverse lookup: {len(reverse_lookup)}")

    # Validate coverage against dataset
    print(f"\n{'=' * 60}")
    print("🔍 COVERAGE VALIDATION")
    print("=" * 60)

    try:
        skills_df = pd.read_csv(SKILLS_CSV, encoding="latin-1")
        dataset_skills = set()
        for entry in skills_df["Skills"].dropna():
            for s in str(entry).split(","):
                cleaned = s.strip().lower()
                if cleaned:
                    dataset_skills.add(cleaned)

        covered = dataset_skills & set(reverse_lookup.keys())
        uncovered = dataset_skills - set(reverse_lookup.keys())

        coverage_pct = len(covered) / len(dataset_skills) * 100 if dataset_skills else 0
        print(f"   Dataset skills:   {len(dataset_skills)}")
        print(f"   Covered by KB:    {len(covered)} ({coverage_pct:.1f}%)")
        print(f"   Not covered:      {len(uncovered)}")

        if uncovered:
            print(f"\n   Uncovered skills (sample):")
            for s in sorted(uncovered)[:20]:
                print(f"      - {s}")
    except FileNotFoundError:
        print("   ⚠️  Skills CSV not found, skipping coverage validation.")

    # Demo: reverse lookup
    print(f"\n{'=' * 60}")
    print("🔗 REVERSE LOOKUP DEMO")
    print("=" * 60)
    demo_skills = ["tensorflow", "python", "docker", "sql", "react", "kubernetes"]
    for skill in demo_skills:
        cats = get_skill_categories(skill, reverse_lookup)
        print(f"   {skill} → {cats}")

    # Demo: category inference
    print(f"\n{'=' * 60}")
    print("🧠 CATEGORY INFERENCE DEMO")
    print("=" * 60)
    sample_resume_skills = ["tensorflow", "python", "pandas", "docker", "sql"]
    inferred = infer_categories(sample_resume_skills, reverse_lookup)
    print(f"   Resume skills: {sample_resume_skills}")
    print(f"   Inferred categories: {inferred}")

    print(f"\n✅ Skill Knowledge Base loaded successfully!")
