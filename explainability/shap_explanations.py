"""
explainability/shap_explanations.py
======================================
SHAP-based explainability for the burnout prediction models.
Provides:
  - Global feature importance (mean |SHAP|)
  - Summary plot (beeswarm)
  - Local/per-student explanation
  - Human-readable reason text (generated from actual SHAP values)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
import importlib.abc
import importlib.machinery

class ShapNumpy2Fix(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'shap.plots.colors._colorconv':
            for finder in sys.meta_path:
                if finder is not self:
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None and spec.loader is not None:
                        orig_loader = spec.loader
                        class PatchedLoader(importlib.abc.Loader):
                            def create_module(self, spec):
                                return orig_loader.create_module(spec)
                            def exec_module(self, module):
                                code = orig_loader.get_source(fullname)
                                patched = code.replace(
                                    'if np.issubdtype(dtype_in, np.dtype(dtype).type):',
                                    'if (isinstance(dtype, type) and np.issubdtype(dtype_in, dtype)) or (not isinstance(dtype, type) and np.issubdtype(dtype_in, np.dtype(dtype).type)):'
                                )
                                exec(compile(patched, spec.origin, 'exec'), module.__dict__)
                        spec.loader = PatchedLoader()
                        return spec
        return None

if ShapNumpy2Fix not in [type(f) for f in sys.meta_path]:
    sys.meta_path.insert(0, ShapNumpy2Fix())

import shap
from sklearn.pipeline import Pipeline

from preprocessing.feature_engineering import FEATURE_COLS

RISK_LABELS = {0: "Low Risk 🟢", 1: "Medium Risk 🟡", 2: "High Risk 🔴"}

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


def _get_clf_from_pipeline(model):
    """Extract the underlying classifier from a sklearn Pipeline."""
    if isinstance(model, Pipeline):
        return model.named_steps["clf"]
    return model


def _get_scaler_from_pipeline(model):
    """Extract the scaler (if any) from a sklearn Pipeline."""
    if isinstance(model, Pipeline) and "scaler" in model.named_steps:
        return model.named_steps["scaler"]
    return None


def _transform_X(model, X: pd.DataFrame) -> np.ndarray:
    """Apply pipeline preprocessing steps to X."""
    scaler = _get_scaler_from_pipeline(model)
    if scaler is not None:
        return scaler.transform(X)
    return X.values


def build_shap_explainer(model, X_train: pd.DataFrame):
    """
    Build a SHAP explainer appropriate for the model type.
    Returns (explainer, X_train_transformed).
    """
    clf = _get_clf_from_pipeline(model)
    X_train_t = _transform_X(model, X_train)

    clf_name = type(clf).__name__
    if clf_name in ("RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier"):
        # Tree-based: use TreeExplainer (fast + exact)
        explainer = shap.TreeExplainer(clf, data=X_train_t)
    else:
        # Linear / black-box: use LinearExplainer or KernelExplainer
        try:
            explainer = shap.LinearExplainer(clf, X_train_t)
        except Exception:
            background = shap.kmeans(X_train_t, 50)
            explainer = shap.KernelExplainer(clf.predict_proba, background)

    return explainer, X_train_t


def compute_shap_values(explainer, X: pd.DataFrame, model) -> np.ndarray:
    """
    Compute SHAP values for a set of samples.
    Returns array of shape (n_samples, n_features, n_classes) or
    (n_samples, n_features) for binary.
    """
    X_t = _transform_X(model, X)
    shap_values = explainer(X_t)
    return shap_values


def global_importance_df(shap_values, feature_names: list = FEATURE_COLS) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature (global importance).
    Works for multi-class (averages across classes).
    """
    vals = shap_values.values  # shape: (n, f) or (n, f, c)
    if vals.ndim == 3:
        # Multi-class: average across classes
        mean_abs = np.abs(vals).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(vals).mean(axis=0)

    df = pd.DataFrame({
        "Feature": feature_names,
        "Description": [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names],
        "Mean |SHAP|": mean_abs,
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    return df


def plot_global_importance(shap_values, feature_names: list = FEATURE_COLS) -> plt.Figure:
    """Bar chart of global SHAP feature importance."""
    df = global_importance_df(shap_values, feature_names)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        df["Description"][::-1],
        df["Mean |SHAP|"][::-1],
        color="#6366f1",
        edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    return fig


def plot_summary(shap_values, X_display: pd.DataFrame, class_idx: int = 2) -> plt.Figure:
    """
    SHAP beeswarm / dot summary plot for a specific target class.
    Default class_idx=2 (High Burnout Risk) to clearly show features driving elevated risk.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = shap_values.values
    if vals.ndim == 3:
        # Multi-class: slice specific target class (0=Low, 1=Medium, 2=High)
        vals_2d = vals[:, :, class_idx]
        plain_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        class_name = plain_labels.get(class_idx, f"Class {class_idx}")
        plot_title = f"SHAP Summary (Feature Impact on {class_name})"
    else:
        vals_2d = vals
        plot_title = "SHAP Summary Plot"

    shap.summary_plot(
        vals_2d,
        X_display,
        feature_names=[FEATURE_DESCRIPTIONS.get(f, f) for f in FEATURE_COLS],
        show=False,
        plot_type="dot",
    )
    fig = plt.gcf()
    plt.title(plot_title, fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def explain_student(
    student_features: pd.DataFrame,
    model,
    explainer,
    target_class: int = None,
    top_n: int = 4,
) -> dict:
    """
    Generate a local SHAP explanation for a single student.

    Parameters
    ----------
    student_features : pd.DataFrame  — shape (1, n_features)
    model            : fitted Pipeline or clf
    explainer        : pre-built SHAP explainer
    target_class     : int (0=Low, 1=Medium, 2=High). If None, explains the predicted class.
    top_n            : number of top factors to return

    Returns
    -------
    dict with keys:
      - prediction         (int 0/1/2)
      - risk_label         (str)
      - target_class       (int)
      - target_class_name  (str)
      - confidence         (float, %)
      - top_factors        (list of dicts)
      - all_factors        (list of dicts)
      - explanation_text   (str, human-readable)
    """
    # Predict
    prediction = int(model.predict(student_features)[0])
    risk_label = RISK_LABELS[prediction]

    # Explicit class selection
    target_class_idx = prediction if target_class is None else int(target_class)
    target_class_name = RISK_LABELS.get(target_class_idx, f"Class {target_class_idx}")

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(student_features)[0]
        confidence = float(proba[prediction] * 100)

    # SHAP values for this student
    X_t = _transform_X(model, student_features)
    sv = explainer(X_t)  # shape: (1, f) or (1, f, c)

    vals = sv.values
    if vals.ndim == 3:
        # Use explicitly selected target class
        class_vals = vals[0, :, target_class_idx]
    else:
        class_vals = vals[0, :]

    # Build factor list with exact class-aware directionality
    factors = []
    for i, (feat, shap_val) in enumerate(zip(FEATURE_COLS, class_vals)):
        raw_val = float(student_features[feat].iloc[0])
        if shap_val > 0:
            direction_desc = f"↑ pushes toward {target_class_name}"
        elif shap_val < 0:
            direction_desc = f"↓ pushes away from {target_class_name}"
        else:
            direction_desc = f"• neutral for {target_class_name}"

        factors.append({
            "feature":      feat,
            "description":  FEATURE_DESCRIPTIONS.get(feat, feat),
            "shap_value":   float(shap_val),
            "raw_value":    raw_val,
            "direction":    direction_desc,
        })

    # Sort by absolute SHAP value
    factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    top_factors = factors[:top_n]

    # Build human-readable explanation text
    lines = [
        f"**Model Predicted Classification: {risk_label}**",
        f"**SHAP Attribution Target:** {target_class_name} (Class {target_class_idx})"
    ]
    if confidence:
        lines.append(f"Model Confidence in {risk_label}: {confidence:.1f}%")
    lines.append(f"\n**Top contributing factors toward {target_class_name}:**")
    for f in top_factors:
        sign = "⬆" if f["shap_value"] > 0 else "⬇"
        lines.append(
            f"- {sign} {f['description']}: `{f['raw_value']:.1f}`  "
            f"(SHAP impact: `{f['shap_value']:+.4f}` → {f['direction']})"
        )

    return {
        "prediction":        prediction,
        "risk_label":        risk_label,
        "target_class":      target_class_idx,
        "target_class_name": target_class_name,
        "confidence":        confidence,
        "top_factors":       top_factors,
        "all_factors":       factors,
        "explanation_text":  "\n".join(lines),
    }
