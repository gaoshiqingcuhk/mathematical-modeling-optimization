#!/usr/bin/env python
"""
Phase 4 standalone runner — loads saved models, regenerates grid if needed.
"""
import sys
import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.config import (
    INPUT_COLUMNS, OUTPUT_COLUMNS, OUTPUT_TABLES, OUTPUT_FIGURES,
    OUTPUT_MODELS, OUTPUT_LOGS, CLEAN_DATA,
)
from src.utils import ensure_dir
from src.data_loader import load_and_clean
from src.robustness import run_phase4
from src.plot_robustness import plot_all_phase4

# Ensure directories
for d in [OUTPUT_LOGS, OUTPUT_TABLES, OUTPUT_FIGURES, OUTPUT_MODELS]:
    ensure_dir(d)

# Load data
print("Loading clean data...")
df = load_and_clean()
print(f"  {len(df)} samples loaded")

# Load trained models
print("\nLoading trained models...")
model_files = {
    "thermal_resistance": ("best_model_y1.joblib", "GPR"),
    "pressure_drop": ("best_model_y2.joblib", "RSM-3-Ridge"),
    "temperature_nonuniformity": ("best_model_y3.joblib", "GPR"),
}
trained_models = {}
for target, (fname, model_name) in model_files.items():
    path = os.path.join(OUTPUT_MODELS, fname)
    pipe = joblib.load(path)
    trained_models[target] = (model_name, pipe)
    print(f"  {target}: {model_name} loaded from {fname}")

# Regenerate surrogate grid (needed for p3_results)
print("\nRegenerating surrogate grid...")
x1_grid = np.linspace(0.10, 0.30, 201)
x2_grid = np.linspace(3.0, 4.5, 151)
x3_grid = np.array([2, 4, 6, 8, 10])
n_candidates = len(x1_grid) * len(x2_grid) * len(x3_grid)
print(f"  Grid: {len(x1_grid)} x {len(x2_grid)} x {len(x3_grid)} = {n_candidates}")

X_cand = []
for x3 in x3_grid:
    for x2 in x2_grid:
        for x1 in x1_grid:
            X_cand.append([x1, x2, x3])
X_cand = np.array(X_cand)

preds = {}
for target in OUTPUT_COLUMNS:
    _, pipe = trained_models[target]
    preds[target] = pipe.predict(X_cand)
    print(f"  Predicted {target}: min={preds[target].min():.6f}, max={preds[target].max():.6f}")

Y_pred = np.column_stack([preds[t] for t in OUTPUT_COLUMNS])

# Compute Pareto for surrogate space
from src.optimization import pareto_frontier
pareto_mask_surr = pareto_frontier(Y_pred)
print(f"  Pareto: {pareto_mask_surr.sum()} candidates")

# Build p3_results with surrogate data
p3_results = {
    "surrogate_results": {
        "X_cand": X_cand,
        "Y_pred": Y_pred,
        "pareto_mask": pareto_mask_surr,
    }
}

# Run Phase 4
print("\n" + "=" * 60)
print("Running Phase 4...")
print("=" * 60)

p4_results = run_phase4(df, p3_results)
plot_all_phase4(p4_results)

print("\n[DONE] Phase 4 completed!")
print(f"  Tables: {OUTPUT_TABLES}")
print(f"  Figures: {OUTPUT_FIGURES}")
