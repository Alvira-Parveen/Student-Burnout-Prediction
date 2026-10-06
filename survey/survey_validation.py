"""
survey/survey_validation.py
============================
Interface for primary survey validation.

PURPOSE
-------
The ML model produces a synthetic proxy burnout score from OULAD.
To validate this scientifically, we collect a real burnout questionnaire
(shortened MBI-GS style) from ~30–50 students, then check whether the
model's predicted risk tier correlates with the self-reported burnout scores.

THIS FILE DOES NOT GENERATE FAKE DATA.
It provides clean functions to load, validate, and analyse REAL survey CSVs
once they are collected.

EXPECTED SURVEY CSV FORMAT
--------------------------
The survey CSV should have the following columns:
  - student_id        : unique identifier (can be anonymised)
  - q1 through q9     : Likert-scale responses (0–6) for MBI-GS short form
  - total_clicks      : from OULAD (student fills in or matched by institution)
  - submission_delay  : from OULAD
  - num_of_prev_attempts : from OULAD
  - studied_credits   : from OULAD

OR — if OULAD features are not directly available from survey respondents:
  The validation can be done by matching student_id against the OULAD
  master dataset (institution provides the mapping).

MBI-GS SHORT FORM (9 questions used here)
------------------------------------------
  Exhaustion subscale (items 1-3):
    q1: "I feel emotionally drained from my studies"
    q2: "I feel used up at the end of a study session"
    q3: "I feel tired when I get up in the morning and have to face another day"
  Cynicism subscale (items 4-6):
    q4: "I have become less interested in my studies since I started"
    q5: "I have become less enthusiastic about my studies"
    q6: "I doubt the significance of my studies"
  Academic Efficacy subscale (items 7-9, REVERSE-scored):
    q7: "I can effectively solve the problems that arise in my studies"
    q8: "I believe that I make an effective contribution to my classes"
    q9: "I feel I am making an effective contribution when I study"
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats


# ─── MBI-GS subscale configuration ───────────────────────────────────────────
MBI_EXHAUSTION_COLS  = ["q1", "q2", "q3"]
MBI_CYNICISM_COLS    = ["q4", "q5", "q6"]
MBI_EFFICACY_COLS    = ["q7", "q8", "q9"]   # reverse-scored
LIKERT_MAX = 6


SURVEY_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "survey_template.csv")


# ─── Template generator ───────────────────────────────────────────────────────
def create_survey_template(output_path: str = SURVEY_TEMPLATE_PATH) -> str:
    """
    Write an empty CSV template that researchers can distribute to students.
    DO NOT fill this with fake data.
    """
    columns = [
        "student_id",
        "q1", "q2", "q3",   # Exhaustion
        "q4", "q5", "q6",   # Cynicism
        "q7", "q8", "q9",   # Efficacy (reverse)
        # Optional OULAD features (if institution can match):
        "total_clicks",
        "submission_delay",
        "num_of_prev_attempts",
        "studied_credits",
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(output_path, index=False)
    print(f"[Survey] Empty template written → {output_path}")
    return output_path


# ─── Loader + validator ───────────────────────────────────────────────────────
def load_survey(path: str) -> pd.DataFrame:
    """Load and validate a filled survey CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Survey CSV not found: {path}\n"
            "Run create_survey_template() to generate the template, "
            "then collect real responses."
        )
    df = pd.read_csv(path)

    required_q = [f"q{i}" for i in range(1, 10)]
    missing = [c for c in required_q + ["student_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"Survey CSV is missing columns: {missing}")

    # Validate Likert range
    for col in required_q:
        invalid = ((df[col] < 0) | (df[col] > LIKERT_MAX)).sum()
        if invalid:
            print(f"  [WARNING] {invalid} invalid values in '{col}' (expected 0–{LIKERT_MAX})")

    print(f"[Survey] Loaded {len(df)} responses from '{path}'")
    return df


# ─── Scoring ──────────────────────────────────────────────────────────────────
def compute_mbi_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MBI-GS subscale scores and composite burnout score.

    Efficacy items are REVERSE-scored (6 - item).

    Composite burnout score formula (common in literature):
        burnout = (exhaustion + cynicism + (6 - efficacy)) / 3
    Higher score → more burnout.
    """
    df = df.copy()

    df["exhaustion_score"] = df[MBI_EXHAUSTION_COLS].mean(axis=1)
    df["cynicism_score"]   = df[MBI_CYNICISM_COLS].mean(axis=1)
    df["efficacy_score"]   = df[MBI_EFFICACY_COLS].mean(axis=1)  # raw

    # Reverse efficacy for composite (higher = worse)
    df["efficacy_score_rev"] = LIKERT_MAX - df["efficacy_score"]

    df["mbi_composite"] = (
        df["exhaustion_score"] + df["cynicism_score"] + df["efficacy_score_rev"]
    ) / 3

    return df


# ─── Validation ───────────────────────────────────────────────────────────────
def validate_model_vs_survey(
    survey_df: pd.DataFrame,
    model,
    feature_cols: list,
    verbose: bool = True,
) -> dict:
    """
    Compare model-predicted burnout risk with MBI composite scores.

    Parameters
    ----------
    survey_df    : DataFrame from load_survey() + compute_mbi_scores()
    model        : fitted sklearn model/Pipeline
    feature_cols : list of feature column names expected by the model
    verbose      : print results

    Returns
    -------
    dict with Spearman correlation, p-value, and comparison DataFrame
    """
    # Check that OULAD features are present
    missing_feats = [f for f in feature_cols if f not in survey_df.columns]
    if missing_feats:
        raise ValueError(
            f"Survey CSV is missing OULAD feature columns needed for prediction: "
            f"{missing_feats}\n"
            "Please match survey respondents to their OULAD records."
        )

    X_survey = survey_df[feature_cols]
    predicted_risk = model.predict(X_survey)

    comparison = survey_df[["student_id", "mbi_composite"]].copy()
    comparison["predicted_risk"] = predicted_risk

    # Spearman rank correlation
    rho, p_val = stats.spearmanr(comparison["mbi_composite"], comparison["predicted_risk"])

    if verbose:
        print("\n[Survey Validation] Results:")
        print(f"  N respondents        : {len(comparison)}")
        print(f"  Spearman ρ           : {rho:.4f}")
        print(f"  p-value              : {p_val:.4f}")
        if p_val < 0.05:
            print("  Interpretation       : Statistically significant correlation (p < 0.05)")
        else:
            print("  Interpretation       : No statistically significant correlation (p ≥ 0.05)")
        print("\n  Predicted risk distribution:")
        print(comparison["predicted_risk"].value_counts().sort_index())

    return {
        "spearman_rho": rho,
        "p_value": p_val,
        "n": len(comparison),
        "comparison_df": comparison,
    }


def plot_validation_scatter(comparison_df: pd.DataFrame) -> plt.Figure:
    """Scatter plot: model predicted risk (x) vs MBI composite (y)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    colors = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}

    for risk_val, group in comparison_df.groupby("predicted_risk"):
        ax.scatter(
            group["predicted_risk"] + np.random.uniform(-0.1, 0.1, len(group)),
            group["mbi_composite"],
            label=risk_labels.get(risk_val, risk_val),
            color=colors.get(risk_val, "#888"),
            alpha=0.7, s=60, edgecolors="white",
        )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low Risk", "Medium Risk", "High Risk"])
    ax.set_ylabel("MBI Composite Score (self-reported)")
    ax.set_title("Model Predicted Risk vs. Self-Reported Burnout (MBI)")
    ax.legend()
    plt.tight_layout()
    return fig
