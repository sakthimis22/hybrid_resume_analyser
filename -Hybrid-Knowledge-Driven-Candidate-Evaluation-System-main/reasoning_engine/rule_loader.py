"""
rule_loader.py

Loads and validates rules from the Rule Knowledge Base (rule_base.json).
Provides functions to access rules by ID, priority, or action type
for use by the reasoning/rule execution engine.
"""

import json
import os

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RULE_BASE_FILE = os.path.join(PROJECT_ROOT, "knowledge_base", "rule_base.json")


def load_rules():
    """Load all rules from the rule knowledge base."""
    with open(RULE_BASE_FILE, "r") as f:
        rule_kb = json.load(f)
    return rule_kb["rules"]


def load_metadata():
    """Load the rule base metadata (version, weights, etc.)."""
    with open(RULE_BASE_FILE, "r") as f:
        rule_kb = json.load(f)
    return rule_kb.get("metadata", {})


def get_rule_by_id(rule_id):
    """Retrieve a specific rule by its ID."""
    for rule in load_rules():
        if rule["rule_id"] == rule_id:
            return rule
    return None


def get_rules_by_action(action_type):
    """Get all rules that perform a specific action (boost, penalize, flag_top)."""
    return [r for r in load_rules() if r["action"] == action_type]


def get_rules_sorted_by_priority():
    """Return rules sorted by priority (lowest number = highest priority)."""
    return sorted(load_rules(), key=lambda r: r["priority"])


# ==================== MAIN ====================
if __name__ == "__main__":

    print("=" * 60)
    print("RULE KNOWLEDGE BASE")
    print("=" * 60)

    # Load metadata
    meta = load_metadata()
    print(f"\nVersion: {meta.get('version', 'unknown')}")
    print(f"Model: {meta.get('scoring_model', 'unknown')}")
    print(f"Base weights: {meta.get('base_weights', {})}")

    # Load and display all rules
    rules = load_rules()
    print(f"\nTotal rules: {len(rules)}")
    print(f"\n{'ID':<6} {'Priority':<10} {'Action':<10} {'Weight':<8} Description")
    print("-" * 60)

    for r in get_rules_sorted_by_priority():
        print(f"{r['rule_id']:<6} {r['priority']:<10} {r['action']:<10} {r['weight']:<8} {r['description']}")

    # Show rules by action type
    print(f"\nBoost rules: {len(get_rules_by_action('boost'))}")
    print(f"Penalize rules: {len(get_rules_by_action('penalize'))}")
    print(f"Flag rules: {len(get_rules_by_action('flag_top'))}")

    print(f"\nRule Knowledge Base loaded successfully!")
