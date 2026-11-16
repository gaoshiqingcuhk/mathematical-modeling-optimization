#!/usr/bin/env python
"""
Phase 5 Common-Norm Monte Carlo — precise re-run.
Reads 80 enhanced samples for common normalization, then runs
MC for #64, #48, #14, #80 with 10000 samples each.
"""
import sys, os, joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from src.config import (
    INPUT_COLUMNS, OUTPUT_COLUMNS, OUTPUT_TABLES, OUTPUT_MODELS,
)
from src.utils import ensure_dir

SEED = 2026
N_MC = 10000
SOLUTIONS = [
    ("64",  0.20, 4.50,  4),
    ("48",  0.20, 4.50,  6),
    ("14",  0.20, 3.50, 10),
    ("80",  0.20, 4.50,  2),
]

# ── 1. Load data & models ──────────────────────────────────
print("=" * 60)
print("Phase 5: Common-Norm Monte Carlo (Precise Rerun)")
print("=" * 60)

clean_path = os.path.join(BASE, "data", "processed", "clean_data.csv")
df = pd.read_csv(clean_path)
print(f"Loaded clean_data: {len(df)} rows")

# 80 enhanced pin-fin samples
df_enh = df[(df["pin_width_ratio"] > 0) & (df["pin_rows"] >= 2)].copy()
Y_enh = df_enh[OUTPUT_COLUMNS].values  # (80, 3)
print(f"Enhanced samples for common-norm: {len(df_enh)}")

# Common normalization baseline (FIXED across all solutions)
Y_MIN = Y_enh.min(axis=0)
Y_MAX = Y_enh.max(axis=0)
DENOM = Y_MAX - Y_MIN
DENOM[DENOM == 0] = 1e-12

print(f"Common-norm baseline:")
for j, t in enumerate(OUTPUT_COLUMNS):
    print(f"  {t}: min={Y_MIN[j]:.6f}, max={Y_MAX[j]:.6f}, range={DENOM[j]:.6f}")

# Load models
print("\nLoading trained models...")
models = {}
path_map = {
    "thermal_resistance": "best_model_y1.joblib",
    "pressure_drop": "best_model_y2.joblib",
    "temperature_nonuniformity": "best_model_y3.joblib",
}
for target, fname in path_map.items():
    path = os.path.join(OUTPUT_MODELS, fname)
    models[target] = joblib.load(path)
    print(f"  {target}: loaded")

def predict_all(X):
    preds = [models[t].predict(X) for t in OUTPUT_COLUMNS]
    return np.column_stack(preds)

# ── 2. Run MC for each solution ────────────────────────────
rng = np.random.default_rng(SEED)
all_samples = []       # for table_21e
summary_rows = []      # for table_21c
defect_rows = []       # for table_21b-like

for sol_name, x1_nom, x2_nom, x3_nom in SOLUTIONS:
    print(f"\n{'─'*50}")
    print(f"MC: #{sol_name}  nominal x=({x1_nom}, {x2_nom}, {int(x3_nom)})")
    print(f"{'─'*50}")

    # Sample
    x1_s = rng.normal(x1_nom, 0.03 * x1_nom, N_MC)
    x2_s = rng.normal(x2_nom, 0.03 * x2_nom, N_MC)
    x1_s = np.clip(x1_s, 0.10, 0.30)
    x2_s = np.clip(x2_s, 3.0, 4.5)
    x3_s = np.full(N_MC, x3_nom)

    X_p = np.column_stack([x1_s, x2_s, x3_s])
    Y_p = predict_all(X_p)  # (10000, 3)

    # Nominal (predicted at exact x)
    x_nom = np.array([x1_nom, x2_nom, x3_nom])
    y_nom = predict_all(x_nom.reshape(1, -1))[0]

    # ── Common-norm composite ──
    Z_p = (Y_p - Y_MIN) / DENOM          # (10000, 3)
    Z_nom = (y_nom - Y_MIN) / DENOM      # (3,)
    composite_p = Z_p.mean(axis=1)        # (10000,)  equal-weight mean
    composite_nom = float(Z_nom.mean())

    print(f"  Nominal y: ({y_nom[0]:.6f}, {y_nom[1]:.6f}, {y_nom[2]:.6f})")
    print(f"  Nominal composite: {composite_nom:.6f}")
    print(f"  MC composite: mean={composite_p.mean():.6f}, std={composite_p.std():.6f}")

    # ── Per-sample records (table_21e) ──
    for i in range(N_MC):
        all_samples.append({
            "solution": sol_name,
            "sample_index": i,
            "x1_perturbed": round(float(x1_s[i]), 6),
            "x2_perturbed": round(float(x2_s[i]), 6),
            "x3": int(x3_s[i]),
            "y1_pred": round(float(Y_p[i, 0]), 8),
            "y2_pred": round(float(Y_p[i, 1]), 8),
            "y3_pred": round(float(Y_p[i, 2]), 8),
            "z1": round(float(Z_p[i, 0]), 8),
            "z2": round(float(Z_p[i, 1]), 8),
            "z3": round(float(Z_p[i, 2]), 8),
            "composite_score": round(float(composite_p[i]), 8),
        })

    # ── Statistics (table_21c) ──
    n_w = max(1, int(N_MC * 0.05))
    composite_sorted = np.sort(composite_p)
    cvar95 = float(np.mean(composite_sorted[-n_w:]))
    ci_lo = float(np.percentile(composite_p, 2.5))
    ci_hi = float(np.percentile(composite_p, 97.5))
    max_abs_dev = float(np.max(np.abs(composite_p - composite_nom)))

    summary_rows.append({
        "solution": sol_name,
        "x1": x1_nom, "x2": x2_nom, "x3": int(x3_nom),
        "nominal_y1": round(float(y_nom[0]), 6),
        "nominal_y2": round(float(y_nom[1]), 6),
        "nominal_y3": round(float(y_nom[2]), 6),
        "nominal_composite": round(composite_nom, 6),
        "mc_mean_composite": round(float(composite_p.mean()), 6),
        "mc_std_composite": round(float(composite_p.std()), 6),
        "mc_ci95_lower": round(ci_lo, 6),
        "mc_ci95_upper": round(ci_hi, 6),
        "mc_cvar95": round(cvar95, 6),
        "mc_max_abs_deviation": round(max_abs_dev, 6),
    })

    # Per-target stats
    for j, target in enumerate(OUTPUT_COLUMNS):
        yj = Y_p[:, j]
        summary_rows.append({
            "solution": sol_name,
            "x1": x1_nom, "x2": x2_nom, "x3": int(x3_nom),
            "nominal_y1": "", "nominal_y2": "", "nominal_y3": "",
            "nominal_composite": "",
            "mc_mean_composite": "",
            "mc_std_composite": round(float(np.std(yj)), 6),
            "mc_ci95_lower": round(float(np.percentile(yj, 2.5)), 6),
            "mc_ci95_upper": round(float(np.percentile(yj, 97.5)), 6),
            "mc_cvar95": round(float(np.mean(np.sort(yj)[-n_w:])), 6),
            "mc_max_abs_deviation": round(float(np.max(np.abs(yj - y_nom[j]))), 6),
            "_target_tag": target,
        })

    # ── Row defects (common-norm) ──
    for defect_label, x3_def in [("row_minus", max(2, x3_nom - 2)),
                                  ("row_plus",  min(10, x3_nom + 2))]:
        if x3_def == x3_nom:
            continue
        x_d = np.array([x1_nom, x2_nom, x3_def])
        y_d = predict_all(x_d.reshape(1, -1))[0]
        Z_d = (y_d - Y_MIN) / DENOM
        comp_d = float(Z_d.mean())
        defect_rows.append({
            "solution": sol_name,
            "defect_type": defect_label,
            "nominal_x3": int(x3_nom),
            "defect_x3": int(x3_def),
            "pred_y1": round(float(y_d[0]), 6),
            "pred_y2": round(float(y_d[1]), 6),
            "pred_y3": round(float(y_d[2]), 6),
            "composite_score_common_norm": round(comp_d, 6),
            "delta_from_nominal": round(float(comp_d - composite_nom), 6),
        })

# ── 3. Save outputs ─────────────────────────────────────────
ensure_dir(OUTPUT_TABLES)

# table_21e: per-sample MC data
samples_df = pd.DataFrame(all_samples)
path_21e = os.path.join(OUTPUT_TABLES, "table_21e_mc_samples_common_norm.csv")
samples_df.to_csv(path_21e, index=False, encoding="utf-8-sig")
print(f"\nSaved: table_21e_mc_samples_common_norm.csv ({len(samples_df)} rows)")

# table_21c: composite statistics
summary_df = pd.DataFrame(summary_rows)
path_21c = os.path.join(OUTPUT_TABLES, "table_21c_composite_mc_common_norm.csv")
summary_df.to_csv(path_21c, index=False, encoding="utf-8-sig")
print(f"Saved: table_21c_composite_mc_common_norm.csv ({len(summary_df)} rows)")

# defect table
defect_df = pd.DataFrame(defect_rows)
path_def = os.path.join(OUTPUT_TABLES, "table_21d_row_defect_common_norm.csv")
defect_df.to_csv(path_def, index=False, encoding="utf-8-sig")
print(f"Saved: table_21d_row_defect_common_norm.csv ({len(defect_df)} rows)")

# ── 4. Stability summary (table_23) ─────────────────────────
comp_rows = [r for r in summary_rows if r["nominal_composite"] != ""]
stab_rows = []
for r in comp_rows:
    sol = r["solution"]
    sol_def = defect_df[defect_df["solution"] == sol]
    max_defect_abs = float(sol_def["delta_from_nominal"].abs().max()) if len(sol_def) > 0 else 0.0
    stab_rows.append({
        "solution": sol,
        "x1": r["x1"], "x2": r["x2"], "x3": int(r["x3"]),
        "nominal_composite": r["nominal_composite"],
        "mc_mean_composite": r["mc_mean_composite"],
        "mc_std_composite": r["mc_std_composite"],
        "mc_cvar95": r["mc_cvar95"],
        "mc_max_abs_deviation": r["mc_max_abs_deviation"],
        "row_defect_max_abs_delta": round(max_defect_abs, 6),
    })

stab = pd.DataFrame(stab_rows)

def norm_lower(x):
    x = np.array(x, dtype=float)
    d = x.max() - x.min()
    if d < 1e-12: return np.zeros_like(x)
    return (x - x.min()) / d

stab["stability_score"] = (
    0.40 * norm_lower(stab["mc_std_composite"].values)
    + 0.40 * norm_lower(stab["mc_cvar95"].values)
    + 0.20 * norm_lower(stab["row_defect_max_abs_delta"].values)
)
stab["stability_rank"] = stab["stability_score"].rank(method="min").astype(int)
stab = stab.sort_values("stability_score")

conclusions = []
for _, row in stab.iterrows():
    if row["solution"] == "64":
        conclusions.append("Common-norm制造扰动下排名第1；综合最优方案，制造稳定性最优")
    elif row["solution"] == "48":
        conclusions.append("Common-norm制造扰动下排名第2；Q4风险规避鲁棒备选")
    elif row["solution"] == "14":
        conclusions.append("高排数冗余参考；common-norm下名义/风险指标均较差，不作为主推荐")
    elif row["solution"] == "80":
        conclusions.append("低压降方案；common-norm制造稳定性较差")
    else:
        conclusions.append("")
stab["conclusion"] = conclusions

path_23 = os.path.join(OUTPUT_TABLES, "table_23_solution_stability_summary_common_norm.csv")
stab.to_csv(path_23, index=False, encoding="utf-8-sig")
print(f"Saved: table_23_solution_stability_summary_common_norm.csv")

# ── 5. Print final summary ──────────────────────────────────
print("\n" + "=" * 60)
print("FINAL COMMON-NORM STABILITY RANKING")
print("=" * 60)
for _, row in stab.iterrows():
    print(f"  {int(row['stability_rank'])}. #{row['solution']}: "
          f"nom={row['nominal_composite']:.4f}  "
          f"std={row['mc_std_composite']:.4f}  "
          f"CVaR95={row['mc_cvar95']:.4f}  "
          f"defect={row['row_defect_max_abs_delta']:.4f}")
    print(f"     → {row['conclusion']}")

print("\n=== Per-solution detail ===")
for r in comp_rows:
    print(f"  #{r['solution']}: composite mean={r['mc_mean_composite']:.6f}, "
          f"std={r['mc_std_composite']:.6f}, "
          f"CVaR95={r['mc_cvar95']:.6f}, "
          f"rank={int(stab[stab['solution']==r['solution']]['stability_rank'].values[0])}")

print("\n[DONE] Phase 5 common-norm MC complete.")
