"""
CANDIDATE CLASSIFIER — candidate_classifier.py
===============================================
Patent Figure 1: After computing the final score, candidates are
classified as TOP_CANDIDATE (score >= 0.5) or STANDARD_CANDIDATE.

This module:
1. Applies the binary classification threshold
2. Generates a classification report
3. Saves a classified output CSV with tier labels
4. Prints a ranked summary table for recruiters
"""

import pandas as pd
import os

TOP_THRESHOLD = 0.5   # Patent Figure 1 explicit threshold


def classify_candidates(
    ranked_csv: str = "outputs/final_ranked_candidates.csv",
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Read ranked candidates, apply tier classification, save results.
    Returns the classified DataFrame.
    """
    df = pd.read_csv(ranked_csv)

    # ── Apply tier classification (Patent Figure 1) ─────────────────────
    df["Tier"] = df["Final_Score"].apply(
        lambda s: "TOP_CANDIDATE" if s >= TOP_THRESHOLD else "STANDARD_CANDIDATE"
    )

    # ── Add rank column ──────────────────────────────────────────────────
    df = df.sort_values("Final_Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)

    # ── Save classified output ───────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "classified_candidates.csv")
    df.to_csv(out_path, index=False)

    # ── Print summary ────────────────────────────────────────────────────
    top      = df[df["Tier"] == "TOP_CANDIDATE"]
    standard = df[df["Tier"] == "STANDARD_CANDIDATE"]

    print("=" * 60)
    print("CANDIDATE TIER CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"Total candidates evaluated : {len(df)}")
    print(f"TOP candidates (score≥0.5) : {len(top)}")
    print(f"STANDARD candidates        : {len(standard)}")
    print(f"Threshold used             : {TOP_THRESHOLD}")
    print(f"Output saved to            : {out_path}")

    if len(top) > 0:
        print("\n--- TOP CANDIDATES ---")
        cols = [c for c in ["Rank", "Resume_ID", "Final_Score", "ML_Score", "Tier"]
                if c in df.columns]
        print(top[cols].head(10).to_string(index=False))

    if len(standard) > 0:
        print("\n--- STANDARD CANDIDATES (bottom 5 shown) ---")
        cols = [c for c in ["Rank", "Resume_ID", "Final_Score", "Tier"] if c in df.columns]
        print(standard[cols].tail(5).to_string(index=False))

    return df


def flag_top_candidates(scores: list, threshold: float = TOP_THRESHOLD) -> list:
    """
    Lightweight helper: classify a list of scores.
    Returns list of tier labels in the same order.
    """
    return ["TOP_CANDIDATE" if s >= threshold else "STANDARD_CANDIDATE" for s in scores]


def classification_stats(df: pd.DataFrame) -> dict:
    """Return summary statistics for the classification run."""
    top = df[df["Tier"] == "TOP_CANDIDATE"]
    std = df[df["Tier"] == "STANDARD_CANDIDATE"]
    return {
        "total":              len(df),
        "top_count":          len(top),
        "standard_count":     len(std),
        "top_avg_score":      round(top["Final_Score"].mean(), 4) if len(top) else 0.0,
        "standard_avg_score": round(std["Final_Score"].mean(), 4) if len(std) else 0.0,
        "top_percentage":     round(len(top) / max(len(df), 1) * 100, 1),
        "threshold_used":     TOP_THRESHOLD,
    }


if __name__ == "__main__":
    classified_df = classify_candidates()
    stats = classification_stats(classified_df)
    print("\n--- CLASSIFICATION STATISTICS ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
