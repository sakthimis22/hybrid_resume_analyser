"""
adaptive_rule_weighting.py

Implements a learning loop for the Rule Knowledge Base.
Analyzes final candidate rankings and "hiring outcomes" to evaluate
how well each rule predicts successfully selected candidates.
Adjusts rule weights dynamically based on this empirical performance.

Updates: knowledge_base/rule_base.json
"""

import pandas as pd
import json
import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

RANKING_FILE = os.path.join(PROJECT_ROOT, "outputs", "hybrid_final_ranking.csv")
RULE_BASE_FILE = os.path.join(PROJECT_ROOT, "knowledge_base", "rule_base.json")


def update_rule_weights():
    print(f"Loading latest ranking data from {RANKING_FILE}...")
    try:
        df = pd.read_csv(RANKING_FILE)
    except FileNotFoundError:
        print("Error: Could not find ranking file. Run the pipeline first.")
        return

    # Simulate hiring decisions: assume the top 100 candidates are "selected/hired"
    # This represents ground truth for our system to learn from
    df["selected"] = df["Rank"] <= 100
    
    rule_performance = {}
    
    # 1. Evaluate Rule Performance
    for _, row in df.iterrows():
        rules_str = str(row.get("Rules_Fired", ""))
        if rules_str and rules_str.lower() != "none" and rules_str != "nan":
            rules = [r.strip() for r in rules_str.split(",")]
            for r in rules:
                if not r:
                    continue
                if r not in rule_performance:
                    rule_performance[r] = {"success": 0, "total": 0}
                    
                rule_performance[r]["total"] += 1
                if row["selected"]:
                    rule_performance[r]["success"] += 1

    # 2. Update Rule Knowledge Base
    print("Loading Rule Knowledge Base...")
    with open(RULE_BASE_FILE, "r") as f:
        rule_kb = json.load(f)
        
    print(f"\nAnalyzing outcomes for {len(df)} candidates ({df['selected'].sum()} selected)...\n")
    print(f"{'Rule ID':<8} {'Fired':<8} {'Success':<8} {'Win Rate':<10} {'Old Wgt':<8} {'New Wgt'}")
    print("-" * 60)
    
    for rule in rule_kb["rules"]:
        rid = rule["rule_id"]
        
        if rid in rule_performance:
            success = rule_performance[rid]["success"]
            total = rule_performance[rid]["total"]
            
            # Win rate = how often a candidate with this rule got hired
            win_rate = success / total if total > 0 else 0
            
            old_weight = rule["weight"]
            
            # Compute new adaptive weight
            if rule["action"] == "penalize" or rule["action"] == "flag_top":
                # Safe fallback: don't automatically adjust penalties or binary flags yet
                new_weight = old_weight
            else:
                # Blend the old weight with the new observed success rate.
                # Max potential learned weight here sets an upper bound (e.g., 0.4 max).
                learned_weight = win_rate * 0.4 
                # Smooth update (momentum) to avoid wild swings
                new_weight = round((old_weight * 0.6) + (learned_weight * 0.4), 3)
                
            rule["weight"] = new_weight
            
            print(f"{rid:<8} {total:<8} {success:<8} {win_rate*100:>5.1f}%     {old_weight:<8} {new_weight}")
        else:
            print(f"{rid:<8} --- Not fired during this cycle ---")

    # Increment KB version string to denote learning update
    current_version = float(rule_kb["metadata"].get("version", "1.0"))
    rule_kb["metadata"]["version"] = str(round(current_version + 0.1, 1))
    rule_kb["metadata"]["last_updated_by"] = "Adaptive Rule Weighting Engine"

    # 3. Save new weights
    with open(RULE_BASE_FILE, "w") as f:
        json.dump(rule_kb, f, indent=4)
        
    print("\n✅ Adaptive Rule Weighting completed.")
    print(f"   knowledge_base/rule_base.json updated to v{rule_kb['metadata']['version']}!")


if __name__ == "__main__":
    update_rule_weights()
