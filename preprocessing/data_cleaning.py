"""
preprocessing/data_cleaning.py
================================
Loads and cleans the OULAD CSVs.
- Reads only required columns from the large studentVle.csv
- Handles duplicates, missing values, and impossible values
- Returns clean per-student-level DataFrames
"""

import os
import pandas as pd
import numpy as np

# ─── Default dataset path (relative to project root) ──────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "DATASET")


def _resolve_path(filename: str) -> str:
    return os.path.join(DATASET_DIR, filename)


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_vle(early_window_days: int | None = None) -> pd.DataFrame:
    """
    Load studentVle.csv efficiently.

    Parameters
    ----------
    early_window_days : int | None
        If set, only load rows where date <= early_window_days.
        Pass None to load all rows (full-semester).

    Returns
    -------
    pd.DataFrame  — columns: id_student, date, sum_click
    """
    path = _resolve_path("studentVle.csv")
    # Read only needed columns to save memory on this 433MB file
    dtype = {"id_student": "int32", "date": "int16", "sum_click": "int32"}
    chunks = []
    chunksize = 500_000
    for chunk in pd.read_csv(
        path,
        usecols=["id_student", "date", "sum_click"],
        dtype=dtype,
        chunksize=chunksize,
    ):
        if early_window_days is not None:
            chunk = chunk[chunk["date"] <= early_window_days]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df


def load_student_assessment() -> pd.DataFrame:
    path = _resolve_path("studentAssessment.csv")
    df = pd.read_csv(
        path,
        dtype={"id_assessment": "int32", "id_student": "int32",
               "date_submitted": "float32", "is_banked": "int8",
               "score": "float32"},
    )
    # Drop banked assessments (carried over from a previous attempt) — not real behaviour
    df = df[df["is_banked"] == 0].copy()
    return df


def load_assessments() -> pd.DataFrame:
    path = _resolve_path("assessments.csv")
    df = pd.read_csv(
        path,
        dtype={"id_assessment": "int32", "date": "float32", "weight": "float32"},
    )
    return df


def load_student_info() -> pd.DataFrame:
    path = _resolve_path("studentInfo.csv")
    df = pd.read_csv(
        path,
        usecols=["id_student", "num_of_prev_attempts", "studied_credits", "final_result"],
        dtype={"id_student": "int32", "num_of_prev_attempts": "int8",
               "studied_credits": "int16"},
    )
    return df


# ─── Cleaning helpers ─────────────────────────────────────────────────────────

def clean_vle_aggregated(vle_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate VLE to one row per student: total clicks.
    Handles duplicates (same student can appear many times — that's expected).
    """
    agg = (
        vle_raw
        .groupby("id_student", as_index=False)["sum_click"]
        .sum()
        .rename(columns={"sum_click": "total_clicks"})
    )
    # Sanity check: no student with negative clicks
    neg = (agg["total_clicks"] < 0).sum()
    if neg:
        print(f"  [WARNING] {neg} students with negative total_clicks — clamping to 0")
        agg["total_clicks"] = agg["total_clicks"].clip(lower=0)
    return agg


def compute_submission_delay(
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
    early_window_days: int | None = None,
) -> pd.DataFrame:
    """
    Merge assessment submissions with deadlines to compute per-student
    average submission delay (days late = positive, early = negative).

    Parameters
    ----------
    early_window_days : int | None
        If set, only include submissions where date_submitted <= early_window_days.
        This prevents leaking future submission behaviour into early-window features.
    """
    merged = pd.merge(
        student_assessment[["id_student", "id_assessment", "date_submitted"]],
        assessments[["id_assessment", "date"]].rename(columns={"date": "deadline"}),
        on="id_assessment",
        how="inner",
    )
    # Filter to early window if requested
    if early_window_days is not None:
        merged = merged[merged["date_submitted"] <= early_window_days].copy()

    merged["submission_delay"] = merged["date_submitted"] - merged["deadline"]

    # Average delay per student
    delay_agg = (
        merged.groupby("id_student", as_index=False)["submission_delay"]
        .mean()
    )
    return delay_agg


def build_clean_master(
    early_window_days: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: load → filter → aggregate → merge → clean.

    Parameters
    ----------
    early_window_days : int | None
        21 for the 3-week early window, None for full-semester baseline.

    Returns
    -------
    pd.DataFrame with one row per student, all raw features, no target column.
    """
    label = f"early-window (≤day {early_window_days})" if early_window_days else "full-semester"
    if verbose:
        print(f"\n[DataCleaning] Building master dataset — mode: {label}")

    # Load
    vle_raw = load_vle(early_window_days=early_window_days)
    sa = load_student_assessment()
    assessments = load_assessments()
    si = load_student_info()

    # Aggregate VLE
    vle_agg = clean_vle_aggregated(vle_raw)

    # Submission delay (respect early window)
    delay_agg = compute_submission_delay(sa, assessments, early_window_days=early_window_days)

    # Merge everything
    df = pd.merge(vle_agg, delay_agg, on="id_student", how="inner")
    df = pd.merge(
        df,
        si[["id_student", "num_of_prev_attempts", "studied_credits"]],
        on="id_student",
        how="inner",
    )

    # ── Data-quality checks ────────────────────────────────────────────────────
    before = len(df)

    # Remove duplicates (shouldn't exist after groupby, but guard anyway)
    df = df.drop_duplicates(subset="id_student")

    # Replace remaining NaNs with column medians (safe imputation)
    for col in ["submission_delay", "num_of_prev_attempts", "studied_credits"]:
        n_missing = df[col].isna().sum()
        if n_missing:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            if verbose:
                print(f"  [INFO] Filled {n_missing} NaN in '{col}' with median={median_val:.2f}")

    # Clamp impossible values
    df["num_of_prev_attempts"] = df["num_of_prev_attempts"].clip(lower=0, upper=10)
    df["studied_credits"] = df["studied_credits"].clip(lower=1)

    after = len(df)
    if verbose:
        print(f"  [INFO] Students after cleaning: {after:,}  (dropped {before - after} duplicates)")
        print(f"  [INFO] total_clicks range: {df['total_clicks'].min():.0f} – {df['total_clicks'].max():.0f}")
        print(f"  [INFO] submission_delay range: {df['submission_delay'].min():.1f} – {df['submission_delay'].max():.1f}")

    return df
