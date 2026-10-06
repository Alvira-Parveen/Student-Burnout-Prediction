"""
models/early_detection_models.py
===================================
Early-window experiment: uses only the first 21 days (3 weeks)
of OULAD student activity to predict burnout risk.

KEY DESIGN DECISIONS
--------------------
1. VLE data filtered to date <= 21 only.
2. Submission delay computed from submissions with date_submitted <= 21 only.
   This prevents any leakage of future submission behaviour.
3. Target labels are still created from the FULL-SEMESTER data
   (total engagement + full delay average), since the GOAL is to
   predict the eventual full-semester outcome from early signals.
   This is correctly documented as a proxy label, not ground truth.
4. Students with NO early-window VLE data are excluded — they have
   insufficient early behaviour to characterise.

Results saved to models/results_early_detection.pkl
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.data_cleaning import build_clean_master
from preprocessing.feature_engineering import (
    engineer_features, create_proxy_labels, get_X_y, FEATURE_COLS
)
from evaluation.model_evaluation import compare_models

RESULTS_PATH     = os.path.join(os.path.dirname(__file__), "results_early_detection.pkl")
BEST_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "best_early_model.pkl")
EARLY_WINDOW_DAYS = 21   # ≈ 3 weeks from module start


def run_early_detection(save: bool = True, verbose: bool = True) -> dict:
    """
    Early-window experiment pipeline.

    Returns
    -------
    dict: comparison_df, fitted_models, X_test, y_test, feature_cols
    """
    # ── 1. Early-window features (input to model) ─────────────────────────────
    if verbose:
        print("\n" + "═" * 60)
        print(f"  EARLY-WINDOW EXPERIMENT  (days 0 – {EARLY_WINDOW_DAYS})")
        print("═" * 60)

    early_master = build_clean_master(
        early_window_days=EARLY_WINDOW_DAYS, verbose=verbose
    )
    early_master = engineer_features(early_master, verbose=verbose)

    # ── 2. Target labels from FULL semester (what we're trying to predict) ────
    # We need full-semester engagement to compute the proxy label
    # (low full-semester engagement = high risk)
    if verbose:
        print("\n  [EarlyDetection] Loading full-semester data to create proxy labels ...")

    full_master = build_clean_master(early_window_days=None, verbose=False)

    # Keep only students also present in early window
    common_students = set(early_master["id_student"]) & set(full_master["id_student"])
    full_for_labels = full_master[full_master["id_student"].isin(common_students)].copy()

    # Create proxy labels from full-semester behaviour
    full_for_labels["burnout_risk_proxy"] = create_proxy_labels(full_for_labels)

    # Attach labels to early features
    label_map = full_for_labels.set_index("id_student")["burnout_risk_proxy"]
    early_master = early_master[early_master["id_student"].isin(common_students)].copy()
    early_master["burnout_risk_proxy"] = early_master["id_student"].map(label_map)
    early_master = early_master.dropna(subset=["burnout_risk_proxy"])
    early_master["burnout_risk_proxy"] = early_master["burnout_risk_proxy"].astype(int)

    if verbose:
        print(f"\n  [EarlyDetection] Students with both early features + labels: "
              f"{len(early_master):,}")
        print("  Class distribution:")
        print(early_master["burnout_risk_proxy"].value_counts().sort_index())

    X, y = get_X_y(early_master)

    # ── 3. Train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 4. Models ─────────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                random_state=42, eval_metric="mlogloss",
                verbosity=0,
            )),
        ]),
    }

    # ── 5. Train + evaluate ───────────────────────────────────────────────────
    comparison_df, fitted_models = compare_models(
        models, X_train, y_train, X_test, y_test,
        cv=5, verbose=verbose,
    )

    if verbose:
        print("\n" + "═" * 60)
        print("  EARLY-DETECTION MODEL COMPARISON")
        print("═" * 60)
        print(comparison_df.to_string())

    best_name  = comparison_df["Test Accuracy"].idxmax()
    best_model = fitted_models[best_name]

    if verbose:
        print(f"\n  [EarlyDetection] Best model: {best_name}  "
              f"(Test Acc: {comparison_df.loc[best_name, 'Test Accuracy']:.4f})")

    result = {
        "comparison_df": comparison_df,
        "fitted_models": fitted_models,
        "best_name": best_name,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": FEATURE_COLS,
        "early_window_days": EARLY_WINDOW_DAYS,
    }

    if save:
        joblib.dump(result, RESULTS_PATH)
        joblib.dump(best_model, BEST_MODEL_PATH)
        if verbose:
            print(f"  [EarlyDetection] Results saved → {RESULTS_PATH}")
            print(f"  [EarlyDetection] Best model saved → {BEST_MODEL_PATH}")

    return result


if __name__ == "__main__":
    run_early_detection(save=True, verbose=True)
