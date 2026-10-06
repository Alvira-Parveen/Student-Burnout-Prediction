"""
preprocessing/feature_engineering.py
======================================
Derives model-ready features from the cleaned master DataFrame.
- Creates 10 behavioural features (same as existing baseline)
- Creates the synthetic burnout-risk proxy label (same formula as baseline)
- Returns X (features) and y (labels) ready for sklearn
"""

import numpy as np
import pandas as pd

# ─── Feature names (ordered — must match what the Streamlit app / model expect) ──
FEATURE_COLS = [
    "total_clicks",        # was sum_click in baseline
    "submission_delay",
    "delay_abs",
    "engagement_level",
    "engagement_per_day",
    "delay_ratio",
    "click_intensity",
    "activity_score",
    "num_of_prev_attempts",
    "studied_credits",
]

FEATURE_DESCRIPTIONS = {
    "total_clicks":         "Platform Engagement (total clicks)",
    "submission_delay":     "Submission Delay (days late/early)",
    "delay_abs":            "Delay Magnitude (absolute days off deadline)",
    "engagement_level":     "Engagement Level (0=Low, 1=Medium, 2=High)",
    "engagement_per_day":   "Engagement per Attempt",
    "delay_ratio":          "Delay Direction Ratio",
    "click_intensity":      "Click Intensity (clicks per credit)",
    "activity_score":       "Activity Score (clicks × engagement level)",
    "num_of_prev_attempts": "Previous Course Attempts",
    "studied_credits":      "Studied Credits (workload)",
}


def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add all derived features to the master DataFrame.
    Input df must contain: total_clicks, submission_delay,
                           num_of_prev_attempts, studied_credits
    """
    df = df.copy()

    # ── Core derived features ────────────────────────────────────────────────
    df["delay_abs"] = df["submission_delay"].abs()

    df["engagement_level"] = pd.cut(
        df["total_clicks"],
        bins=[0, 1_000, 3_000, np.inf],
        labels=[0, 1, 2],
        right=True,
    ).astype(float)
    df["engagement_level"] = df["engagement_level"].fillna(0)

    # clicks per attempt (avoid zero-division)
    df["engagement_per_day"] = df["total_clicks"] / (df["num_of_prev_attempts"] + 1)

    # directional delay ratio (-1 to +1-ish)
    df["delay_ratio"] = df["submission_delay"] / (df["delay_abs"] + 1)

    # clicks relative to workload (credit-normalised engagement)
    df["click_intensity"] = df["total_clicks"] / (df["studied_credits"] + 1)

    # combined engagement signal
    df["activity_score"] = df["total_clicks"] * df["engagement_level"]

    if verbose:
        print(f"  [FeatureEng] Features ready — shape: {df[FEATURE_COLS].shape}")

    return df


def create_proxy_labels(df: pd.DataFrame, random_seed: int = 42) -> pd.Series:
    """
    Create the synthetic burnout-risk proxy label using the same formula
    as the existing baseline.

    IMPORTANT — This is a PROXY label, NOT real burnout ground-truth.
    The formula uses relative percentile ranks so:
      - Low engagement (relative to peers) → higher burnout score
      - Later submission (relative to peers) → higher burnout score
    Students are then split into 3 equal quantile classes.

    Returns
    -------
    pd.Series  — integer 0 / 1 / 2 (Low / Medium / High proxy risk)
    """
    rng = np.random.default_rng(random_seed)
    score = (
        df["total_clicks"].rank(pct=True) * -1
        + df["submission_delay"].rank(pct=True)
        + rng.uniform(0, 0.3, size=len(df))  # tie-breaking noise
    )
    labels = pd.qcut(score, q=3, labels=[0, 1, 2]).astype(int)
    return labels.rename("burnout_risk_proxy")


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and proxy-label vector y from a fully
    engineered DataFrame.
    """
    X = df[FEATURE_COLS].copy()
    y = df["burnout_risk_proxy"].copy()
    return X, y
