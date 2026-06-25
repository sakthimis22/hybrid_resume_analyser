"""
ultimate_hybrid_ranking.py

The final integration layer of the intelligent recruitment system.
Combines all three reasoning engines:
1. ML Similarity Score (from TF-IDF + Cosine)
2. Rule Execution Score (from Rule Knowledge Base)
3. Graph Skill Score (from Semantic Graph Distance)

Formula:
  Ultimate_Score = (0.5 * ML) + (0.3 * Rules) + (0.2 * Graph)

Output: outputs/ultimate_ranked_candidates.csv
"""

import pandas as pd
import ast
import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from knowledge_graph.graph_skill_reasoning import graph_skill_score
from ranking_engine.skill_matcher import skill_kb

INPUT_RULES = os.path.join(PROJECT_ROOT, "outputs", "rule_engine_results.csv")
ENTITIES_FILE = os.path.join(PROJECT_ROOT, "datasets", "processed", "resume_entities.csv")
RAW_JOB_FILE = os.path.join(PROJECT_ROOT, "datasets", "raw", "job_descriptions", "job_descriptions.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "ultimate_ranked_candidates.csv")

WEIGHT_ML = 0.5
WEIGHT_RULES = 0.3
WEIGHT_GRAPH = 0.2


def get_job_skills():
    """Extract job skills using the taxonomy (same logic as rule_based_ranking)."""
    try:
        raw_job_df = pd.read_csv(RAW_JOB_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        raw_job_df = pd.read_csv(RAW_JOB_FILE, encoding="latin-1")
        
    raw_skills = str(raw_job_df["skills"].iloc[0]) if "skills" in raw_job_df.columns else ""
    raw_resp = str(raw_job_df["Responsibilities"].iloc[0]) if "Responsibilities" in raw_job_df.columns else ""
    job_text = (raw_skills + " " + raw_resp).lower()
    
    job_skills_list = []
    for category, skills in skill_kb.items():
        for skill in skills:
            if skill.lower() in job_text:
                job_skills_list.append(skill.lower())
    return list(set(job_skills_list))


def ultimate_ranking():
    print("Loading candidate data and extracting job requirements...")
    
    try:
        df = pd.read_csv(INPUT_RULES)
        entities_df = pd.read_csv(ENTITIES_FILE)
    except FileNotFoundError:
        print("Error: Required input files not found. Run earlier pipeline stages first.")
        return

    job_skills = get_job_skills()
    print(f"Target Job Skills ({len(job_skills)}): {job_skills[:5]}...")
    
    ultimate_scores = []
    graph_scores_list = []
    
    total = len(df)
    print("Computing Graph Skill Scores and merging all reasoning layers...")
    
    for idx, row in df.iterrows():
        resume_id = int(row["Resume_ID"])
        
        # Get ML and Rule Scores
        ml_score = float(row.get("ML_Score", 0))
        rule_score = float(row.get("Rule_Score", 0))
        
        # Extract resume skills
        if resume_id < len(entities_df):
            skills_str = entities_df.iloc[resume_id]["skills_extracted"]
            try:
                resume_skills = ast.literal_eval(skills_str) if isinstance(skills_str, str) else []
            except:
                resume_skills = []
        else:
            resume_skills = []
            
        # 3. Compute Graph Skill Reasoning score
        graph_score = graph_skill_score(resume_skills, job_skills)
        graph_scores_list.append(round(graph_score, 4))
        
        # Final Ultimate Hybrid Formula
        ultimate_score = (
            (WEIGHT_ML * ml_score) + 
            (WEIGHT_RULES * rule_score) + 
            (WEIGHT_GRAPH * graph_score)
        )
        
        ultimate_scores.append(round(ultimate_score, 4))
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total} candidates...")

    df["Graph_Score"] = graph_scores_list
    df["Ultimate_Score"] = ultimate_scores
    
    # Sort candidates by Ultimate Score
    df = df.sort_values(by="Ultimate_Score", ascending=False)
    
    # Reorder columns to showcase the 3 intelligence pillars
    cols = ["Resume_ID", "Ultimate_Score", "ML_Score", "Rule_Score", "Graph_Score", "Rules_Fired", "Rule_Details"]
    
    # Bring in any additional columns that were in the dataframe safely
    for c in df.columns:
        if c not in cols and c not in ["Rank", "Combined_Score", "Hybrid_Score"]:
            cols.append(c)
            
    df = df[cols]
    df.insert(0, "Final_Rank", range(1, len(df) + 1))
    
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'=' * 60}")
    print("🚀 ULTIMATE HYBRID RANKING COMPLETED")
    print(f"{'=' * 60}")
    print(f"Formula: ({WEIGHT_ML} * ML) + ({WEIGHT_RULES} * Rule) + ({WEIGHT_GRAPH} * Graph)")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total ranked: {len(df)}")
    
    print(f"\n🏆 Top 5 Intelligent Matches:\n")
    for _, r in df.head(5).iterrows():
        print(f"  Rank #{int(r['Final_Rank'])} (ID:{int(r['Resume_ID'])}) — Ultimate Score: {r['Ultimate_Score']:.4f}")
        print(f"    - ML Similarity: {r['ML_Score']:.4f}")
        print(f"    - Rule Engine:   {r['Rule_Score']:.4f}")
        print(f"    - Graph Logic:   {r['Graph_Score']:.4f}")
        print()


if __name__ == "__main__":
    ultimate_ranking()
