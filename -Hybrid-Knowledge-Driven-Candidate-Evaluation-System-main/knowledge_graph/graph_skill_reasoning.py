"""
graph_skill_reasoning.py

Graph-Based Skill Reasoning Engine.
Computes a skill match score using graph relationships.
Instead of requiring exact text matches, this engine uses the
Skill Knowledge Graph to find shortest paths.

For example:
  Resume has "tensorflow"
  Job requires "machine learning"
  Graph: tensorflow → Machine Learning (Distance = 1)
  Score awarded: 0.8
"""

import networkx as nx
import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from knowledge_graph.skill_graph_builder import build_skill_graph

# Cache the graph so we don't rebuild it on every call
_GRAPH = None

def load_graph():
    global _GRAPH
    if _GRAPH is None:
        # We suppress the print statements from the builder during reasoning
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        _GRAPH = build_skill_graph()
        sys.stdout = sys.__stdout__
    return _GRAPH


def graph_skill_score(resume_skills, job_skills):
    """
    Computes a match score based on graph distance.
    Exact match: 1.0 points
    Distance 1 : 0.8 points (e.g., skill -> direct category)
    Distance 2 : 0.6 points (e.g., skill -> domain)
    """
    G = load_graph()
    
    if not job_skills:
        return 0.0

    resume_skills_lower = [s.lower().strip() for s in resume_skills]
    job_skills_lower = [s.lower().strip() for s in job_skills]

    total_score = 0.0
    total_requirements = len(job_skills_lower)

    for job_skill in job_skills_lower:
        best_match_for_this_req = 0.0

        for resume_skill in resume_skills_lower:
            
            # 1. Exact Match
            if resume_skill == job_skill:
                best_match_for_this_req = max(best_match_for_this_req, 1.0)
                if best_match_for_this_req == 1.0:
                    break  # Maximum possible score for this requirement

            # 2. Graph Reasoning (Inference Match)
            if resume_skill in G.nodes and job_skill in G.nodes:
                try:
                    # Search for path from specific resume skill to broader job requirement
                    # e.g. resume="tensorflow" -> job="machine learning"
                    distance = nx.shortest_path_length(G, source=resume_skill, target=job_skill)
                    
                    if distance == 1:
                        best_match_for_this_req = max(best_match_for_this_req, 0.8)
                    elif distance == 2:
                        best_match_for_this_req = max(best_match_for_this_req, 0.6)
                        
                except nx.NetworkXNoPath:
                    pass

        total_score += best_match_for_this_req

    return total_score / total_requirements if total_requirements > 0 else 0.0


# ==================== MAIN ====================
if __name__ == "__main__":

    print("=" * 60)
    print("GRAPH-BASED SKILL REASONING ENGINE")
    print("=" * 60)

    # Test 1: Exact match + 1-hop inference
    resume_1 = ["tensorflow", "python"]
    job_1 = ["machine learning", "python"]
    score_1 = graph_skill_score(resume_1, job_1)
    
    print(f"\nTest 1 (Exact + 1-hop):")
    print(f"  Resume: {resume_1}")
    print(f"  Job:    {job_1}")
    print(f"  Graph Score: {score_1:.2f}")
    print(f"  Reasoning: python(1.0) + tensorflow->machine learning(0.8) = 1.8 / 2 = 0.90")

    # Test 2: 2-hop inference
    resume_2 = ["react", "docker"]
    job_2 = ["software engineering", "cloud computing"]
    score_2 = graph_skill_score(resume_2, job_2)
    
    print(f"\nTest 2 (2-hop Deep Inference):")
    print(f"  Resume: {resume_2}")
    print(f"  Job:    {job_2}")
    print(f"  Graph Score: {score_2:.2f}")
    print(f"  Reasoning: react->web dev->software eng(0.6) + docker->devops->cloud(0.6) = 1.2 / 2 = 0.60")

    # Test 3: Unrelated skills
    resume_3 = ["photoshop", "illustrator"]
    job_3 = ["machine learning", "python"]
    score_3 = graph_skill_score(resume_3, job_3)
    
    print(f"\nTest 3 (Unrelated):")
    print(f"  Resume: {resume_3}")
    print(f"  Job:    {job_3}")
    print(f"  Graph Score: {score_3:.2f}")
    
    print(f"\n{'=' * 60}")
    print("Graph Skill Reasoning Engine is active!")
