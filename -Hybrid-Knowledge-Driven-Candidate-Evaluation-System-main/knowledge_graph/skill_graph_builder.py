"""
skill_graph_builder.py

Builds a NetworkX Directed Graph (DiGraph) from the skill taxonomy.
This converts the hierarchical JSON knowledge base into a Graph-Based
Skill Reasoning Engine that enables multi-level inference (distance
computation between skills, roles, and domains).
"""

import json
import networkx as nx
import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

TAXONOMY_FILE = os.path.join(PROJECT_ROOT, "knowledge_base", "skill_taxonomy.json")


def build_skill_graph():
    """Builds and returns a Directed Graph of skills and categories."""
    with open(TAXONOMY_FILE, "r") as f:
        taxonomy = json.load(f)

    G = nx.DiGraph()

    # Base Level 1: Taxonomy categories to individual skills
    for category, skills in taxonomy.items():
        # Add category node
        G.add_node(category, type="category")
        
        for skill in skills:
            # Add skill node
            skill_lower = skill.lower().strip()
            G.add_node(skill_lower, type="skill")
            
            # The edge points from skill -> category (inference direction)
            G.add_edge(skill_lower, category)

    # Level 2: Define higher-order domain relationships
    # This matches the inference engine hierarchy
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

    # Add higher order domains
    for child_cat, parent_cat in CATEGORY_HIERARCHY.items():
        if child_cat in G.nodes:
            G.add_node(parent_cat, type="domain")
            G.add_edge(child_cat, parent_cat)

    print("✅ Skill graph created.")
    print(f"Nodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")

    return G


if __name__ == "__main__":
    
    print("=" * 60)
    print("SKILL KNOWLEDGE GRAPH BUILDER")
    print("=" * 60)
    
    graph = build_skill_graph()

    print(f"\nExample Edges (Skill -> Category / Domain):")
    print("-" * 60)
    
    # Show a mix of edges
    edges_to_show = list(graph.edges)[:5]
    for edge in edges_to_show:
        print(f"{edge[0]}  →  {edge[1]}")
        
    # Specifically show some interesting hierarchy paths
    print("\nDemonstrating deep inference paths:")
    try:
        paths = list(nx.all_simple_paths(graph, source="tensorflow", target="Artificial Intelligence"))
        for path in paths:
            print(f" {' → '.join(path)}")
            
        paths = list(nx.all_simple_paths(graph, source="react", target="Software Engineering"))
        for path in paths:
            print(f" {' → '.join(path)}")
    except nx.NetworkXNoPath:
        print("Path not found for examples.")
        
    print(f"\n{'=' * 60}")
    print("Graph builder is ready for the Knowledge Reasoning Engine!")
