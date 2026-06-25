"""
rule_execution_engine.py

Evaluates candidate data against the Rule Knowledge Base.
For each candidate:
  1. Loads all rules from rule_base.json
  2. Evaluates each rule's condition against the candidate's scores
  3. Fires matching rules and accumulates the rule score
  4. Records which rules fired for explainability

Output: outputs/rule_engine_results.csv
"""

import pandas as pd
import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from reasoning_engine.rule_loader import load_rules, load_metadata

INPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "final_ranked_candidates.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "outputs", "rule_engine_results.csv")


def evaluate_condition(condition, candidate):
    """
    Safely evaluate a rule condition against candidate data.

    Args:
        condition: String expression (e.g., "skill_score >= 0.5")
        candidate: Dict of candidate score variables.

    Returns:
        bool: Whether the condition is satisfied.
    """
    try:
        return eval(condition, {"__builtins__": {}}, candidate)
    except Exception:
        return False


def execute_rules(candidate):
    """
    Execute all rules against a single candidate.

    Returns:
        tuple: (rule_score, fired_rules, fired_details)
    """
    rules = load_rules()

    rule_score = 0.0
    fired_rules = []
    fired_details = []

    for rule in rules:
        if evaluate_condition(rule["condition"], candidate):
            rule_score += rule["weight"]
            fired_rules.append(rule["rule_id"])
            fired_details.append(
                f"{rule['rule_id']}:{rule['description']}({rule['action']} {rule['weight']:+.2f})"
            )

    return rule_score, fired_rules, fired_details


def run_rule_engine():
    """Run the rule execution engine on all ranked candidates."""

    print("Loading ranked candidates...")
    df = pd.read_csv(INPUT_FILE)

    meta = load_metadata()
    rules = load_rules()
    print(f"Rule KB v{meta.get('version', '?')} loaded — {len(rules)} rules")

    rule_scores = []
    rule_ids_fired = []
    rule_details = []
    top_flags = []

    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):

        # Map CSV columns to rule variable names
        candidate = {
            "ml_score": float(row.get("ML_Score", 0)),
            "skill_score": float(row.get("Skill_Score", 0)),
            "exp_score": float(row.get("Exp_Score", 0)),
            "final_score": float(row.get("Final_Score", 0)),
        }

        score, fired, details = execute_rules(candidate)

        rule_scores.append(round(score, 4))
        rule_ids_fired.append(",".join(fired) if fired else "none")
        rule_details.append(" | ".join(details) if details else "no rules fired")
        top_flags.append("TOP" if "R9" in fired else "")

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total} candidates...")

    df["Rule_Score"] = rule_scores
    df["Rules_Fired"] = rule_ids_fired
    df["Rule_Details"] = rule_details
    df["Top_Flag"] = top_flags

    # Sort by combined final + rule score
    df["Combined_Score"] = (df["Final_Score"] + df["Rule_Score"]).round(4)
    df = df.sort_values(by="Combined_Score", ascending=False)
    df["Rank"] = range(1, len(df) + 1)

    # Reorder columns
    df = df[["Rank", "Resume_ID", "ML_Score", "Skill_Score", "Exp_Score",
             "Final_Score", "Rule_Score", "Combined_Score", "Top_Flag",
             "Rules_Fired", "Rule_Details"]]

    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("Rule Execution Engine completed!")
    print(f"{'=' * 60}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total candidates: {len(df)}")
    print(f"Top-flagged candidates: {(df['Top_Flag'] == 'TOP').sum()}")

    print(f"\nTop 5 candidates:\n")
    for _, r in df.head(5).iterrows():
        print(f"  Rank #{int(r['Rank'])} (ID:{int(r['Resume_ID'])}) "
              f"Combined:{r['Combined_Score']} "
              f"[ML:{r['ML_Score']} Skill:{r['Skill_Score']} Exp:{r['Exp_Score']}]")
        print(f"    Rules fired: {r['Rules_Fired']}")
        print()


if __name__ == "__main__":
    run_rule_engine()
