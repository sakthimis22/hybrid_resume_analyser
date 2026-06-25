"""
integration_pipeline.py

The master pipeline script that connects dynamically uploaded resumes 
to the full Intelligent Hybrid Architecture:
  1. Skill Extraction
  2. Machine Learning Similarity
  3. Rule Engine Execution
  4. Graph-Based Skill Reasoning
  5. Ultimate Hybrid Ranking
"""

import pandas as pd
import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from resume_parser.extract_resume_entities import extract_skills, extract_experience
from knowledge_graph.graph_skill_reasoning import graph_skill_score
from pipeline.resume_ranking_pipeline import run_ranking_pipeline
from reasoning_engine.rule_execution_engine import execute_rules
from resume_analysis.section_analyzer import analyze_resume_sections
from knowledge_representation.resume_knowledge_builder import build_resume_knowledge
from explainability.reasoning_trace_generator import generate_reasoning_trace
import json

UPLOAD_TEXTS_FILE = os.path.join(PROJECT_ROOT, "outputs", "uploaded_resume_texts.csv")
RAW_JOB_FILE = os.path.join(PROJECT_ROOT, "datasets", "raw", "job_descriptions", "job_descriptions.csv")
RANKS_FILE = os.path.join(PROJECT_ROOT, "outputs", "uploaded_resume_rankings.csv")
FINAL_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "ultimate_uploaded_rankings.csv")

# Weights for Ultimate Fusion
WEIGHT_ML = 0.5
WEIGHT_RULES = 0.3
WEIGHT_GRAPH = 0.2


def get_job_skills_and_text(target_role=None):
    """
    Get the raw job text and the required skills for a specific role.
    Loads from knowledge_base/job_roles.json for high-signal text generation.
    """
    import json
    role_file = os.path.join(PROJECT_ROOT, "knowledge_base", "job_roles.json")
    
    if os.path.exists(role_file):
        with open(role_file, "r") as f:
            roles = json.load(f)
            
        if target_role and target_role in roles:
            skills = roles[target_role]
            job_text = f"{target_role} " + " ".join(skills)
            return job_text.lower(), [s.lower() for s in skills]

    # Fallback to first row of CSV dataset
    try:
        jobs_df = pd.read_csv(RAW_JOB_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        jobs_df = pd.read_csv(RAW_JOB_FILE, encoding="latin-1")
        
    job_row = jobs_df.iloc[0] if not jobs_df.empty else None
    if job_row is None:
        raise ValueError("Job dataset is empty.")
            
    raw_skills = str(job_row.get("skills", ""))
    raw_resp = str(job_row.get("Responsibilities", ""))
    job_text = (raw_skills + " " + raw_resp).lower()
    
    job_skills_list = []
    if raw_skills and raw_skills.lower() != "nan":
        job_skills_list = [s.strip().lower() for s in raw_skills.split(",")]
        
    return job_text, job_skills_list


def run_full_pipeline(target_role="AI Engineer", custom_job_text=None, custom_job_skills=None):
    print(f"\n========================================================")
    print(f"🚀 INTEGRATION PIPELINE: Target Role [{target_role}]")
    print(f"========================================================")

    # 1. Load uploaded texts
    try:
        resumes_df = pd.read_csv(UPLOAD_TEXTS_FILE)
    except FileNotFoundError:
        print("Error: No resumes extracted. Upload resumes and run the analyzer first.")
        return

    if custom_job_text is not None and custom_job_skills is not None:
        job_text = custom_job_text.lower()
        job_skills = custom_job_skills
        print(f"Using Custom Job Description with {len(job_skills)} extracted skills.")
    else:
        job_text, job_skills = get_job_skills_and_text(target_role)
        print(f"Found {len(job_skills)} target job skills for {target_role}.")

    # 2. Run initial ML Similarity (computes "ML_Score" and creates rankings file)
    print("\nExecuting Machine Learning Engine...")
    run_ranking_pipeline(job_text_override=job_text) 
    
    # Reload the dataframe now that it has ML_Score
    ranked_df = pd.read_csv(RANKS_FILE)
    
    # 3. Process the 3 Pillars of Intelligence
    print("Executing Skill Extraction, Graph Reasoning, and Rule Engine...")
    
    integrated_results = []
    all_knowledge = []
    all_traces = []
    
    for _, row in ranked_df.iterrows():
        file_name = row["file_name"]
        resume_text = str(row["resume_text"])
        ml_score = float(row["ML_Score"])
        
        # A. Knowledge Representation Layer (Build formal Knowledge Entity)
        knowledge = build_resume_knowledge(file_name, resume_text)
        all_knowledge.append(knowledge)
        
        # Use formal Knowledge Entity elements for the rest of reasoning
        resume_skills = knowledge["skills"]
        sections = analyze_resume_sections(resume_text) # Keep raw text for Streamlit UI
        
        # B. Experience Extraction (for rule engine)
        exp_str = extract_experience(resume_text)
        try:
            exp_val = float(exp_str.split()[0])
        except:
            exp_val = 0.0
            
        # C. Graph Reasoning Score
        graph_score = graph_skill_score(resume_skills, job_skills)
        
        # D. Structural Section Weighting (Simulates Seniority/Depth)
        experience_score = min(1.0, len(knowledge["experience"]) / 500.0)
        projects_score = min(1.0, len(knowledge["projects"]) / 500.0)
        
        # Pre-compute partial skills score for rule engine
        skill_score = 0.0
        if job_skills:
            matched = set([s.lower() for s in resume_skills]).intersection(set([s.lower() for s in job_skills]))
            skill_score = len(matched) / len(job_skills)
            
        # E. Rule Execution Engine
        candidate_metrics = {
            "ml_score": ml_score,
            "skill_score": skill_score,
            "exp_score": 1.0 if exp_val > 0 else 0.0,
            "final_score": 0.0
        }
        
        rule_score, rules_fired, rule_details = execute_rules(candidate_metrics)
        
        # G. Ultimate Hybrid Fusion (Clamped at 0.0)
        ultimate_score = max(0.0, (
            (WEIGHT_ML * ml_score) + 
            (WEIGHT_RULES * rule_score) + 
            (WEIGHT_GRAPH * graph_score) + 
            (0.1 * experience_score) + 
            (0.1 * projects_score)
        ))
        
        # H. Missing Skills Analysis
        matched_skills_list = list(set([s.lower() for s in resume_skills]).intersection(set([s.lower() for s in job_skills])))
        missing_skills = list(set([s.lower() for s in job_skills]) - set([s.lower() for s in resume_skills]))
        
        # I. Generate and Record Reasoning Trace
        trace = generate_reasoning_trace(
            candidate=file_name,
            matched_skills=matched_skills_list,
            missing_skills=missing_skills,
            rules_fired=rules_fired,
            ml_score=ml_score,
            rule_score=rule_score,
            graph_score=graph_score,
            hybrid_score=ultimate_score
        )
        all_traces.append(trace)
        
        # J. Serialize section outputs to pass to the UI
        sections_json = json.dumps(sections)
        
        integrated_results.append({
            "Candidate": file_name,
            "Extracted_Skills": ", ".join(resume_skills),
            "Missing_Skills": ", ".join(missing_skills) if missing_skills else "None",
            "Rules_Triggered": ",".join(rules_fired) if rules_fired else "none",
            "ML_Score": round(ml_score, 4),
            "Graph_Score": round(graph_score, 4),
            "Rule_Score": round(rule_score, 4),
            "Exp_Weight": round(experience_score * 0.1, 4),
            "Proj_Weight": round(projects_score * 0.1, 4),
            "Final_Score": round(ultimate_score, 4),
            "Sections": sections_json
        })

    # Save final structured outputs
    final_df = pd.DataFrame(integrated_results)
    
    # Sort
    final_df = final_df.sort_values(by="Final_Score", ascending=False)
    final_df.insert(0, "Rank", range(1, len(final_df) + 1))
    
    # Overwrite the original ML ranking output with our powerful detailed data for Streamlit
    final_df.to_csv(FINAL_OUTPUT_FILE, index=False)
    
    # Save the formal structured Knowledge Entities globally
    knowledge_path = os.path.join(PROJECT_ROOT, "outputs", "resume_knowledge.json")
    with open(knowledge_path, 'w') as f:
        json.dump(all_knowledge, f, indent=4)
        
    # Save the reasoning traces globally
    traces_path = os.path.join(PROJECT_ROOT, "outputs", "reasoning_traces.txt")
    with open(traces_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_traces))
    
    print("\n========================================================")
    print("✅ FULL HYBRID PIPELINE COMPLETED")
    print("========================================================")
    for _, r in final_df.head(5).iterrows():
        print(f"#{r['Rank']} {r['Candidate']:<20} | Final: {r['Final_Score']:.4f} [ML:{r['ML_Score']} Graph:{r['Graph_Score']} Rule:{r['Rule_Score']}]")

if __name__ == "__main__":
    run_full_pipeline("AI Engineer")
