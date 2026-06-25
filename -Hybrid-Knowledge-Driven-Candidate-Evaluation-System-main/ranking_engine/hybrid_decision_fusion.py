"""
hybrid_decision_fusion.py

Combines ML Similarity Scores and Rule Execution Scores into a
single final hybrid score, representing the ultimate system decision.

Output: outputs/hybrid_final_ranking.csv
"""

import pandas as pd
import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

INPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "rule_engine_results.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "hybrid_final_ranking.csv")

# Weights for final fusion
WEIGHT_ML = 0.6
WEIGHT_RULES = 0.4


def hybrid_fusion():
    print("Loading rule engine results...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run rule_execution_engine.py first.")
        return

    # In rule_execution_engine.py, we already created Combined_Score
    # But this module formally calculates standard Hybrid_Score
    
    hybrid_scores = []
    
    for _, row in df.iterrows():
        # Get base ML score
        ml_score = float(row.get("ML_Score", 0))
        # Get rule score (which encapsulates skill, exp, and composite rules)
        rule_score = float(row.get("Rule_Score", 0))
        
        # Hybrid fusion formula
        hybrid_score = (WEIGHT_ML * ml_score) + (WEIGHT_RULES * rule_score)
        hybrid_scores.append(round(hybrid_score, 4))
        
    df["Hybrid_Score"] = hybrid_scores
    
    # Sort candidates by the new Hybrid Score
    df = df.sort_values(by="Hybrid_Score", ascending=False)
    
    # Update Rank
    df["Rank"] = range(1, len(df) + 1)
    
    # Reorder columns to feature the new Hybrid_Score
    cols = list(df.columns)
    # Move Hybrid_Score right after Rule_Score
    if "Hybrid_Score" in cols:
        cols.remove("Hybrid_Score")
        rule_score_idx = cols.index("Rule_Score") if "Rule_Score" in cols else len(cols)
        cols.insert(rule_score_idx + 1, "Hybrid_Score")
        df = df[cols]

    # Drop the old 'Combined_Score' if it exists to avoid confusion
    if "Combined_Score" in df.columns:
        df = df.drop(columns=["Combined_Score"])
        
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'=' * 60}")
    print("Hybrid Decision Fusion Engine completed!")
    print(f"{'=' * 60}")
    print(f"Formula: ({WEIGHT_ML} * ML_Score) + ({WEIGHT_RULES} * Rule_Score)")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total candidates: {len(df)}")
    
    print(f"\nTop 5 Candidates (Hybrid Ranking):\n")
    for _, r in df.head(5).iterrows():
        print(f"  Rank #{int(r['Rank'])} (ID:{int(r['Resume_ID'])}) — Hybrid Score: {r['Hybrid_Score']:.4f}")
        print(f"    ML Contribution:   {r['ML_Score']:.4f} * {WEIGHT_ML} = {r['ML_Score'] * WEIGHT_ML:.4f}")
        print(f"    Rule Contribution: {r['Rule_Score']:.4f} * {WEIGHT_RULES} = {r['Rule_Score'] * WEIGHT_RULES:.4f}")
        print(f"    Rules Fired:       {r['Rules_Fired']}")
        print()


if __name__ == "__main__":
    hybrid_fusion()
