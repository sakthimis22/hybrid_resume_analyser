import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from resume_analysis.section_analyzer import analyze_resume_sections
from resume_parser.extract_resume_entities import extract_skills

def build_resume_knowledge(candidate_name, resume_text):
    """
    Transforms unstructured resume text into a structured formal Knowledge Unit.
    This creates the explicit Knowledge Representation Layer needed for advanced reasoning.
    """
    sections = analyze_resume_sections(resume_text)

    # Clean skill extraction from relevant sections
    focused_text = sections.get("skills", "") + " " + sections.get("projects", "") + " " + sections.get("experience", "")
    skills = extract_skills(focused_text)

    knowledge = {
        "candidate": candidate_name,
        "education": sections.get("education", ""),
        "experience": sections.get("experience", ""),
        "projects": sections.get("projects", ""),
        "skills": skills,
        "certifications": sections.get("certifications", "")
    }

    return knowledge
