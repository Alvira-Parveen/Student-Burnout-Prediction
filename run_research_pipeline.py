"""
run_research_pipeline.py
=========================
Trains both baseline and early-detection models and saves results.
Run once: python3 run_research_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("  STUDENT BURNOUT — RESEARCH PIPELINE")
print("=" * 60)

from models.baseline_models import run_baseline
from models.early_detection_models import run_early_detection

print("\n[STEP 1] Running BASELINE (full-semester) experiment...")
baseline_result = run_baseline(save=True, verbose=True)

print("\n\n[STEP 2] Running EARLY-DETECTION (first 21 days) experiment...")
early_result = run_early_detection(save=True, verbose=True)

print("\n\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"\nBaseline best model  : {baseline_result['best_name']}")
bl_acc = baseline_result['comparison_df'].loc[baseline_result['best_name'], 'Test Accuracy']
print(f"  Test Accuracy      : {bl_acc:.4f}")

print(f"\nEarly-detection best : {early_result['best_name']}")
ed_acc = early_result['comparison_df'].loc[early_result['best_name'], 'Test Accuracy']
print(f"  Test Accuracy      : {ed_acc:.4f}")

delta = bl_acc - ed_acc
print(f"\nAccuracy trade-off   : {delta:+.4f}  (cost of predicting 3 weeks early)")
print("\nModels saved in models/ directory.")
print("Run: streamlit run app.py")
