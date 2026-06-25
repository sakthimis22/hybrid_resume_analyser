"""
RULE EVALUATION ENGINE — adaptive_rule_weighting.py
====================================================
Patent Claim 7: The adaptive rule weighting mechanism is configured to
dynamically adjust weighting factors associated with evaluation rules
based on candidate evaluation outcomes.

TRUE adaptive weighting:
- Starts with default weights
- Updates weights based on recruiter feedback (hired/rejected outcomes)
- Persists weights to disk so they improve over time
"""

import json
import os


DEFAULT_WEIGHTS = {
    "mandatory_skill_match_rule":  0.30,
    "min_experience_rule":         0.25,
    "programming_language_rule":   0.20,
    "ml_domain_knowledge_rule":    0.15,
    "data_skills_rule":            0.10,
}

WEIGHTS_FILE  = "outputs/adaptive_rule_weights.json"
LEARNING_RATE = 0.05
MIN_WEIGHT    = 0.05
MAX_WEIGHT    = 0.50


def load_weights() -> dict:
    """Load persisted weights, or return defaults on first run."""
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE) as f:
            return json.load(f)
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict) -> None:
    os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)


def normalize_weights(weights: dict) -> dict:
    """Force weights to sum to 1.0."""
    total = sum(weights.values())
    return {k: round(v / total, 6) for k, v in weights.items()}


def compute_weighted_rule_score(activated_rules: list, weights: dict = None) -> float:
    """
    Score a candidate using current adaptive weights.
    Only rules that actually fired contribute their weight.
    """
    if weights is None:
        weights = load_weights()
    score = sum(weights.get(rule, 0.0) for rule in activated_rules)
    return round(min(score, 1.0), 4)


def update_weights_from_feedback(feedback_records: list, weights: dict = None) -> dict:
    """
    Patent Claim 7 — Adaptive update loop.

    feedback_records: list of dicts:
        {"activated_rules": [...], "outcome": 1 or 0}
        outcome=1 means hired/good match; 0 means rejected.

    Rules consistently appearing in successful hires gain weight.
    Rules consistently appearing in rejections lose weight.
    """
    if weights is None:
        weights = load_weights()

    rule_signal = {rule: 0.0 for rule in weights}
    rule_count  = {rule: 0   for rule in weights}

    for record in feedback_records:
        outcome = record.get("outcome", 0)
        for rule in record.get("activated_rules", []):
            if rule in rule_signal:
                rule_signal[rule] += (1 if outcome == 1 else -1)
                rule_count[rule]  += 1

    updated = dict(weights)
    for rule in updated:
        if rule_count[rule] > 0:
            avg_signal = rule_signal[rule] / rule_count[rule]
            updated[rule] += LEARNING_RATE * avg_signal
            updated[rule] = max(MIN_WEIGHT, min(MAX_WEIGHT, updated[rule]))

    updated = normalize_weights(updated)
    save_weights(updated)
    print("Adaptive weights updated and saved.")
    return updated


def get_weight_report(weights: dict = None) -> str:
    if weights is None:
        weights = load_weights()
    lines = ["=== CURRENT ADAPTIVE RULE WEIGHTS ==="]
    for rule, w in sorted(weights.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        lines.append(f"  {rule:<38} {w:.4f}  {bar}")
    lines.append(f"\n  TOTAL: {sum(weights.values()):.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import random
    random.seed(42)

    feedback = []
    for _ in range(10):
        hired = random.random() > 0.4
        rules = []
        if hired:
            rules += ["mandatory_skill_match_rule", "min_experience_rule"]
            if random.random() > 0.5:
                rules.append("ml_domain_knowledge_rule")
        else:
            if random.random() > 0.5:
                rules.append("programming_language_rule")
            if random.random() > 0.5:
                rules.append("data_skills_rule")
        feedback.append({"activated_rules": rules, "outcome": int(hired)})

    print("Before update:")
    print(get_weight_report())

    updated = update_weights_from_feedback(feedback)
    print("\nAfter update (10 feedback records):")
    print(get_weight_report(updated))

    score = compute_weighted_rule_score(
        ["mandatory_skill_match_rule", "min_experience_rule", "ml_domain_knowledge_rule"],
        updated
    )
    print(f"\nSample adaptive rule score: {score}")
