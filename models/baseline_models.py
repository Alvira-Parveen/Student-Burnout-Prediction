"""
models/baseline_models.py
===========================
Baseline experiment using FULL-SEMESTER OULAD data.
Replicates the existing notebook logic but with:
  - Proper Pipeline (no data leakage from scaler)
  - All 4 models compared
  - Results saved to models/results_baseline.pkl
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

# Project-level imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.data_cleaning import build_clean_master
from preprocessing.feature_engineering import engineer_features, create_proxy_labels, get_X_y, FEATURE_COLS
from evaluation.model_evaluation import compare_models, plot_confusion_matrix

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results_baseline.pkl")
BEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_baseline_model.pkl")


def run_baseline(save: bool = True, verbose: bool = True) -> dict:
    """
    Full baseline experiment pipeline.

    Returns
    -------
    dict with keys: comparison_df, fitted_models, X_test, y_test
    """
    # ── 1. Data ───────────────────────────────────────────────────────────────
    master = build_clean_master(early_window_days=None, verbose=verbose)
    master = engineer_features(master, verbose=verbose)
    master["burnout_risk_proxy"] = create_proxy_labels(master)

    if verbose:
        print("\n  [Baseline] Class distribution:")
        print(master["burnout_risk_proxy"].value_counts().sort_index())

    X, y = get_X_y(master)

    # ── 2. Train/test split ───────────────────────────────────────────────────
    # Scaler is fit ONLY on train data via Pipeline → no leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 3. Define models (each wrapped in Pipeline to prevent leakage) ────────
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

    # ── 4. Train + evaluate all models ───────────────────────────────────────
    comparison_df, fitted_models = compare_models(
        models, X_train, y_train, X_test, y_test,
        cv=5, verbose=verbose,
    )

    if verbose:
        print("\n" + "═" * 60)
        print("  BASELINE MODEL COMPARISON")
        print("═" * 60)
        print(comparison_df.to_string())

    # ── 5. Pick best model (by Test Accuracy) ────────────────────────────────
    best_name = comparison_df["Test Accuracy"].idxmax()
    best_model = fitted_models[best_name]

    if verbose:
        print(f"\n  [Baseline] Best model: {best_name}  "
              f"(Test Acc: {comparison_df.loc[best_name, 'Test Accuracy']:.4f})")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    result = {
        "comparison_df": comparison_df,
        "fitted_models": fitted_models,
        "best_name": best_name,
        "best_model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": FEATURE_COLS,
    }

    if save:
        joblib.dump(result, RESULTS_PATH)
        joblib.dump(best_model, BEST_MODEL_PATH)
        if verbose:
            print(f"  [Baseline] Results saved → {RESULTS_PATH}")
            print(f"  [Baseline] Best model saved → {BEST_MODEL_PATH}")

    return result


if __name__ == "__main__":
    run_baseline(save=True, verbose=True)
