"""
evaluation/model_evaluation.py
================================
Cross-validation, metrics, confusion matrix, and ROC-AUC utilities.
Used by both baseline and early-detection experiments.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize


LABEL_NAMES = ["Low Risk", "Medium Risk", "High Risk"]


def cross_validate_model(model, X, y, cv: int = 5, verbose: bool = True) -> dict:
    """
    Run stratified k-fold CV and return mean ± std of key metrics.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    cv_results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    results = {}
    for metric in scoring:
        key = f"test_{metric}"
        results[metric] = {
            "mean": cv_results[key].mean(),
            "std":  cv_results[key].std(),
        }

    if verbose:
        name = type(model).__name__
        print(f"\n  [{name}] {cv}-fold CV results:")
        for m, vals in results.items():
            print(f"    {m:20s}: {vals['mean']:.4f} ± {vals['std']:.4f}")

    return results


def evaluate_on_test(model, X_test, y_test, verbose: bool = True) -> dict:
    """
    Evaluate a fitted model on held-out test data.
    Returns accuracy, precision, recall, f1, roc_auc, and classification report.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # ROC-AUC (OvR, macro) — requires predict_proba
    roc_auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        roc_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")

    results = {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "roc_auc_macro": roc_auc,
    }

    if verbose:
        name = type(model).__name__
        print(f"\n  [{name}] Test-set results:")
        for k, v in results.items():
            if v is not None:
                print(f"    {k:22s}: {v:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=LABEL_NAMES)}")

    return results


def compare_models(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    cv: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train + evaluate all models. Returns a comparison DataFrame.

    Parameters
    ----------
    models : dict  {name_str: unfitted_model_object}
    """
    rows = []
    fitted = {}

    for name, model in models.items():
        if verbose:
            print(f"\n{'─'*50}")
            print(f"  Training: {name}")

        model.fit(X_train, y_train)
        fitted[name] = model

        test_metrics = evaluate_on_test(model, X_test, y_test, verbose=verbose)
        cv_metrics   = cross_validate_model(model, X_train, y_train, cv=cv, verbose=verbose)

        row = {"Model": name}
        row["Test Accuracy"]    = test_metrics["accuracy"]
        row["Test F1 (macro)"]  = test_metrics["f1_macro"]
        row["Test ROC-AUC"]     = test_metrics["roc_auc_macro"]
        row["CV Acc (mean)"]    = cv_metrics["accuracy"]["mean"]
        row["CV Acc (std)"]     = cv_metrics["accuracy"]["std"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df, fitted


def plot_confusion_matrix(model, X_test, y_test, title: str = "") -> plt.Figure:
    """Return a matplotlib Figure with the confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title or f"Confusion Matrix — {type(model).__name__}")
    plt.tight_layout()
    return fig
