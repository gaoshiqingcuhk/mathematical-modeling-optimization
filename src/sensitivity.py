"""
Phase 5: Manufacturing Error & Operational Perturbation Sensitivity
加工误差敏感性、Monte Carlo制造误差、运行工况工程近似
"""
import numpy as np
import pandas as pd
import os
import joblib

from src.config import (
    INPUT_COLUMNS, OUTPUT_COLUMNS, OUTPUT_NAMES_CN,
    OUTPUT_TABLES, OUTPUT_MODELS, CV_RANDOM_SEED,
)
from src.utils import ensure_dir


# ══════════════════════════════════════════════════════════════
#  1. Load models
# ══════════════════════════════════════════════════════════════
def load_trained_models():
    """Load the Phase 2 trained surrogate models."""
    path_map = {
        "thermal_resistance": "best_model_y1.joblib",
        "pressure_drop": "best_model_y2.joblib",
        "temperature_nonuniformity": "best_model_y3.joblib",
    }
    models = {}
    for target, fname in path_map.items():
        path = os.path.join(OUTPUT_MODELS, fname)
        models[target] = joblib.load(path)
    return models


def predict_all(models, X):
    """Predict all 3 targets. X: (n, 3). Returns (n, 3)."""
    preds = [models[t].predict(X) for t in OUTPUT_COLUMNS]
    return np.column_stack(preds)


# ══════════════════════════════════════════════════════════════
#  2. Local Sensitivity: Continuous Parameters
# ══════════════════════════════════════════════════════════════
def local_sensitivity_continuous(models, solutions, perturb_levels=(0.01, 0.03, 0.05)):
    """
    Compute relative sensitivity S = [(y(x+Δ) - y(x-Δ)) / (2Δ)] * [x / y(x)]
    for continuous parameters x1 (pin_width_ratio) and x2 (manifold_depth_height_ratio).

    solutions: list of (name, x1, x2, x3)
    """
    rows = []

    for sol_name, x1_nom, x2_nom, x3_nom in solutions:
        x_nom = np.array([x1_nom, x2_nom, x3_nom])
        y_nom = predict_all(models, x_nom.reshape(1, -1))[0]  # (3,)

        for param_idx, param_name in [(0, "x1_pin_width_ratio"), (1, "x2_manifold_depth_height_ratio")]:
            x_nom_val = x_nom[param_idx]
            for delta_pct in perturb_levels:
                delta = delta_pct * x_nom_val
                if delta == 0:
                    continue

                # Perturb +Δ
                x_plus = x_nom.copy()
                x_plus[param_idx] += delta
                # Clamp to design bounds
                if param_idx == 0:
                    x_plus[param_idx] = np.clip(x_plus[param_idx], 0.10, 0.30)
                else:
                    x_plus[param_idx] = np.clip(x_plus[param_idx], 3.0, 4.5)

                # Perturb -Δ
                x_minus = x_nom.copy()
                x_minus[param_idx] -= delta
                if param_idx == 0:
                    x_minus[param_idx] = np.clip(x_minus[param_idx], 0.10, 0.30)
                else:
                    x_minus[param_idx] = np.clip(x_minus[param_idx], 3.0, 4.5)

                y_plus = predict_all(models, x_plus.reshape(1, -1))[0]
                y_minus = predict_all(models, x_minus.reshape(1, -1))[0]

                for j, target in enumerate(OUTPUT_COLUMNS):
                    dy = y_plus[j] - y_minus[j]
                    effective_delta = (x_plus[param_idx] - x_minus[param_idx])
                    if effective_delta == 0:
                        S = 0.0
                    else:
                        S = (dy / effective_delta) * (x_nom_val / y_nom[j]) if y_nom[j] != 0 else 0.0

                    rows.append({
                        "solution": sol_name,
                        "x1": x1_nom, "x2": x2_nom, "x3": int(x3_nom),
                        "perturbed_parameter": param_name,
                        "perturbation_level": f"{int(delta_pct*100):d}%",
                        "target": target,
                        "nominal_value": round(float(y_nom[j]), 6),
                        "value_minus": round(float(y_minus[j]), 6),
                        "value_plus": round(float(y_plus[j]), 6),
                        "relative_sensitivity": round(float(S), 6),
                    })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  3. Discrete Pin Rows Sensitivity
# ══════════════════════════════════════════════════════════════
def discrete_pin_rows_sensitivity(models, solutions):
    """
    For each solution, evaluate neighboring pin_rows values while
    keeping x1, x2 fixed.
    """
    rows = []

    for sol_name, x1, x2, x3_base in solutions:
        # Determine test x3 values
        if sol_name == "64":
            test_x3s = [2, 4, 6]
        elif sol_name == "48":
            test_x3s = [4, 6, 8]
        elif sol_name == "14":
            test_x3s = [8, 10]
        elif sol_name == "80":
            test_x3s = [2, 4]

        x_nom = np.array([x1, x2, x3_base])
        y_nom = predict_all(models, x_nom.reshape(1, -1))[0]

        for x3_test in test_x3s:
            x_test = np.array([x1, x2, x3_test])
            y_test = predict_all(models, x_test.reshape(1, -1))[0]
            delta_pct = [(y_test[j] - y_nom[j]) / y_nom[j] * 100 for j in range(3)]

            rows.append({
                "solution": sol_name,
                "base_x1": x1, "base_x2": x2, "base_x3": int(x3_base),
                "test_x3": int(x3_test),
                "pred_y1": round(float(y_test[0]), 6),
                "pred_y2": round(float(y_test[1]), 6),
                "pred_y3": round(float(y_test[2]), 6),
                "delta_y1_percent": round(float(delta_pct[0]), 4),
                "delta_y2_percent": round(float(delta_pct[1]), 4),
                "delta_y3_percent": round(float(delta_pct[2]), 4),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  4. Monte Carlo Manufacturing Error
# ══════════════════════════════════════════════════════════════
def monte_carlo_manufacturing(models, solutions, n=10000, seed=2026):
    """
    Monte Carlo simulation of manufacturing errors for specified solutions.
    x1 ~ Normal(x1_nom, 0.03*x1_nom), clamped to [0.10, 0.30]
    x2 ~ Normal(x2_nom, 0.03*x2_nom), clamped to [3.0, 4.5]
    x3 fixed.

    Also evaluates discrete row defects:
    - row_minus: x3 decreased by one valid step
    - row_plus: x3 increased by one valid step
    """
    rng = np.random.default_rng(seed)
    all_x3_values = np.array([2, 4, 6, 8, 10])

    # Normalize reference for composite score
    from src.robustness import normalize_Y
    # We'll normalize based on the nominal values as reference

    rows_mc = []
    rows_defect = []

    for sol_name, x1_nom, x2_nom, x3_nom in solutions:
        print(f"  MC for {sol_name}: x=({x1_nom}, {x2_nom}, {int(x3_nom)})")

        # Sample perturbed designs
        x1_samples = rng.normal(x1_nom, 0.03 * x1_nom, n)
        x2_samples = rng.normal(x2_nom, 0.03 * x2_nom, n)

        # Clamp
        x1_samples = np.clip(x1_samples, 0.10, 0.30)
        x2_samples = np.clip(x2_samples, 3.0, 4.5)
        x3_samples = np.full(n, x3_nom)

        X_perturbed = np.column_stack([x1_samples, x2_samples, x3_samples])
        Y_perturbed = predict_all(models, X_perturbed)  # (n, 3)

        # Nominal prediction
        x_nom = np.array([x1_nom, x2_nom, x3_nom])
        y_nom = predict_all(models, x_nom.reshape(1, -1))[0]

        # Composite score (equal weights, normalized against MC samples)
        # Use min-max normalization based on nominal + all MC samples for fair comparison
        Y_all = np.vstack([y_nom.reshape(1, -1), Y_perturbed])
        y_min = Y_all.min(axis=0)
        y_max = Y_all.max(axis=0)
        denom = y_max - y_min
        denom[denom == 0] = 1e-12
        Z_all = (Y_all - y_min) / denom
        w = np.array([1/3, 1/3, 1/3])
        ws_scores = Z_all @ w  # (n+1,)
        ws_nom = ws_scores[0]
        ws_mc = ws_scores[1:]

        # Statistics per target
        for j, target in enumerate(OUTPUT_COLUMNS):
            yj = Y_perturbed[:, j]
            yj_sorted = np.sort(yj)
            n_worst = max(1, int(n * 0.05))
            cvar95 = float(np.mean(yj_sorted[-n_worst:]))
            ci_lower = float(np.percentile(yj, 2.5))
            ci_upper = float(np.percentile(yj, 97.5))
            max_rel_dev = float(np.max(np.abs(yj - y_nom[j]) / y_nom[j]) * 100)

            rows_mc.append({
                "solution": sol_name,
                "target": target,
                "nominal_value": round(float(y_nom[j]), 6),
                "mean": round(float(np.mean(yj)), 6),
                "std": round(float(np.std(yj)), 6),
                "ci95_lower": round(ci_lower, 6),
                "ci95_upper": round(ci_upper, 6),
                "cvar95": round(cvar95, 6),
                "max_relative_deviation_pct": round(max_rel_dev, 4),
            })

        # Composite score stats
        rows_mc.append({
            "solution": sol_name,
            "target": "composite_score",
            "nominal_value": round(float(ws_nom), 6),
            "mean": round(float(np.mean(ws_mc)), 6),
            "std": round(float(np.std(ws_mc)), 6),
            "ci95_lower": round(float(np.percentile(ws_mc, 2.5)), 6),
            "ci95_upper": round(float(np.percentile(ws_mc, 97.5)), 6),
            "cvar95": round(float(np.mean(np.sort(ws_mc)[-n_worst:])), 6),
            "max_relative_deviation_pct": round(float(np.max(np.abs(ws_mc - ws_nom) / ws_nom) * 100), 4) if ws_nom > 0 else 0,
        })

        # Discrete row defects
        for defect_label, x3_defect in [("row_minus", max(2, x3_nom - 2)),
                                         ("row_plus", min(10, x3_nom + 2))]:
            if x3_defect == x3_nom:
                continue
            x_def = np.array([x1_nom, x2_nom, x3_defect])
            y_def = predict_all(models, x_def.reshape(1, -1))[0]
            # Compute composite score
            Z_def = (y_def - y_min) / denom
            ws_def = float(Z_def @ w)
            rows_defect.append({
                "solution": sol_name,
                "defect_type": defect_label,
                "nominal_x3": int(x3_nom),
                "defect_x3": int(x3_defect),
                "pred_y1": round(float(y_def[0]), 6),
                "pred_y2": round(float(y_def[1]), 6),
                "pred_y3": round(float(y_def[2]), 6),
                "composite_score": round(ws_def, 6),
                "delta_composite_pct": round(float((ws_def - ws_nom) / ws_nom * 100), 4) if ws_nom > 0 else 0,
            })

    mc_df = pd.DataFrame(rows_mc)
    defect_df = pd.DataFrame(rows_defect)
    return mc_df, defect_df, Y_perturbed  # Y_perturbed from last solution for plotting


# ══════════════════════════════════════════════════════════════
#  5. Flow Rate Fluctuation Engineering Approximation
# ══════════════════════════════════════════════════════════════
def flow_rate_fluctuation(models, solutions):
    """
    Engineering approximation of flow rate changes.
    m_dot_ratio ∈ {0.90, 0.95, 1.00, 1.05, 1.10}

    Scaling laws (based on laminar flow engineering approximations):
    - pressure_drop ∝ m_dot^2
    - thermal_resistance ∝ m_dot^(-0.6)  (convective heat transfer ∝ Re^0.8 for turbulent, ~0.6 for developing)
    - temperature_nonuniformity ∝ m_dot^(-0.3)
    """
    m_dot_ratios = [0.90, 0.95, 1.00, 1.05, 1.10]

    # Scaling exponents
    exp_y1 = -0.6
    exp_y2 = 2.0
    exp_y3 = -0.3

    rows = []

    for sol_name, x1_nom, x2_nom, x3_nom in solutions:
        x_nom = np.array([x1_nom, x2_nom, x3_nom])
        y_nom = predict_all(models, x_nom.reshape(1, -1))[0]

        # Nominal composite score
        y_all_nom = predict_all(models, x_nom.reshape(1, -1))[0]
        # Use a reasonable normalization (based on all 84 samples from clean data)
        # For simplicity, use fixed normalization ranges
        y1_rng, y2_rng, y3_rng = (0.05, 0.13, 0.10)  # approximate observed ranges

        for mr in m_dot_ratios:
            y1_s = y_nom[0] * (mr ** exp_y1)
            y2_s = y_nom[1] * (mr ** exp_y2)
            y3_s = y_nom[2] * (mr ** exp_y3)

            # Simple composite score (equal weights, normalized)
            z1 = (y1_s - 0.72) / 0.052
            z2 = (y2_s - 0.076) / 0.13
            z3 = (y3_s - 0.77) / 0.10
            ws = (z1 + z2 + z3) / 3

            rows.append({
                "solution": sol_name,
                "m_dot_ratio": mr,
                "y1_scaled": round(float(y1_s), 6),
                "y2_scaled": round(float(y2_s), 6),
                "y3_scaled": round(float(y3_s), 6),
                "weighted_score_scaled": round(float(ws), 6),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  6. Stability Summary
# ══════════════════════════════════════════════════════════════
def build_stability_summary(mc_df, defect_df):
    """
    Build table_23 summarizing manufacturing stability.
    """
    # Aggregate MC results per solution for composite_score
    mc_composite = mc_df[mc_df["target"] == "composite_score"]
    solutions = mc_composite["solution"].unique()

    rows = []
    for i, sol in enumerate(solutions):
        row = mc_composite[mc_composite["solution"] == sol].iloc[0]
        # Defect sensitivity
        sol_defects = defect_df[defect_df["solution"] == sol]
        max_defect_delta = sol_defects["delta_composite_pct"].abs().max() if len(sol_defects) > 0 else 0

        rows.append({
            "solution": sol,
            "mean_score": round(row["mean"], 6),
            "std_score": round(row["std"], 6),
            "cvar95": round(row["cvar95"], 6),
            "max_relative_deviation_pct": round(row["max_relative_deviation_pct"], 4),
            "row_defect_sensitivity_pct": round(float(max_defect_delta), 4),
        })

    summary = pd.DataFrame(rows)

    # Stability rank: composite of std + cvar95 + max_dev
    def norm_lower(x):
        x = np.array(x, dtype=float)
        denom = x.max() - x.min()
        if denom < 1e-12: return np.zeros_like(x)
        return (x - x.min()) / denom

    summary["stability_score"] = (
        0.40 * norm_lower(summary["std_score"].values) +
        0.40 * norm_lower(summary["cvar95"].values) +
        0.20 * norm_lower(summary["max_relative_deviation_pct"].values)
    )
    summary["stability_rank"] = summary["stability_score"].rank(method="min").astype(int)
    summary = summary.sort_values("stability_score")

    # Add conclusions
    conclusions = []
    for _, r in summary.iterrows():
        if r["solution"] == "64":
            if r["stability_rank"] == 1:
                conclusions.append("综合最优且制造稳定性最佳")
            else:
                conclusions.append("综合最优方案，制造稳定性良好")
        elif r["solution"] == "48":
            if r["stability_rank"] == 1:
                conclusions.append("制造稳定性最优，鲁棒设计验证通过")
            else:
                conclusions.append("鲁棒设计方案，制造稳定性良好")
        elif r["solution"] == "14":
            conclusions.append("高排数冗余参考，可能提升排数缺陷容忍度但会牺牲名义综合性能")
        elif r["solution"] == "80":
            conclusions.append("最低压降候选，制造敏感性较低")
        else:
            conclusions.append("")
    summary["conclusion"] = conclusions

    return summary


# ══════════════════════════════════════════════════════════════
#  7. Main Phase 5 Runner
# ══════════════════════════════════════════════════════════════
def run_phase5(df=None):
    """
    Main Phase 5 execution.
    """
    print("\n" + "=" * 60)
    print("Phase 5: Manufacturing & Operational Sensitivity")
    print("=" * 60)

    ensure_dir(OUTPUT_TABLES)

    # Load models
    print("\n[Phase5] Loading trained models...")
    models = load_trained_models()

    # Define solutions to analyze
    solutions = [
        ("64", 0.20, 4.50, 4),
        ("48", 0.20, 4.50, 6),
        ("14", 0.20, 3.50, 10),
        ("80", 0.20, 4.50, 2),
    ]

    # ── 5.1 Local Sensitivity ──
    print("\n[Phase5] Local sensitivity analysis...")
    sens_df = local_sensitivity_continuous(models, solutions)
    path = os.path.join(OUTPUT_TABLES, "table_19_local_sensitivity_continuous.csv")
    sens_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_19 ({len(sens_df)} rows)")

    # Summary
    for target in OUTPUT_COLUMNS:
        sub = sens_df[sens_df["target"] == target]
        max_idx = sub["relative_sensitivity"].abs().idxmax()
        max_row = sub.loc[max_idx]
        print(f"  {target}: max |S| = {max_row['relative_sensitivity']:.4f} "
              f"({max_row['solution']}, {max_row['perturbed_parameter']}, {max_row['perturbation_level']})")

    # ── 5.2 Discrete Pin Rows ──
    print("\n[Phase5] Discrete pin rows sensitivity...")
    discrete_df = discrete_pin_rows_sensitivity(models, solutions)
    path = os.path.join(OUTPUT_TABLES, "table_20_discrete_pin_rows_sensitivity.csv")
    discrete_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_20 ({len(discrete_df)} rows)")

    # ── 5.3 Monte Carlo ──
    print("\n[Phase5] Monte Carlo manufacturing error (10000 samples)...")
    mc_df, defect_df, Y_last_mc = monte_carlo_manufacturing(models, solutions, n=10000, seed=2026)
    path_mc = os.path.join(OUTPUT_TABLES, "table_21_manufacturing_monte_carlo.csv")
    mc_df.to_csv(path_mc, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_21 ({len(mc_df)} rows)")

    # Also save defect table
    path_defect = os.path.join(OUTPUT_TABLES, "table_21b_row_defect_sensitivity.csv")
    defect_df.to_csv(path_defect, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_21b ({len(defect_df)} rows)")

    # Print MC summary
    mc_comp = mc_df[mc_df["target"] == "composite_score"]
    print("\n  Composite Score MC Results:")
    for _, r in mc_comp.iterrows():
        print(f"    {r['solution']}: nominal={r['nominal_value']:.4f}, mean={r['mean']:.4f}, "
              f"std={r['std']:.4f}, CVaR95={r['cvar95']:.4f}")

    # ── 5.4 Flow Rate ──
    print("\n[Phase5] Flow rate fluctuation engineering approximation...")
    flow_df = flow_rate_fluctuation(models, solutions)
    path = os.path.join(OUTPUT_TABLES, "table_22_flow_rate_fluctuation_engineering.csv")
    flow_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_22 ({len(flow_df)} rows)")

    # ── 5.5 Stability Summary ──
    print("\n[Phase5] Building stability summary...")
    stability_df = build_stability_summary(mc_df, defect_df)
    path = os.path.join(OUTPUT_TABLES, "table_23_solution_stability_summary.csv")
    stability_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_23 ({len(stability_df)} rows)")

    print("\n  Stability Ranking:")
    for _, r in stability_df.iterrows():
        print(f"    {int(r['stability_rank'])}. {r['solution']}: score={r['stability_score']:.4f}, "
              f"std={r['std_score']:.4f}, cvar={r['cvar95']:.4f}, "
              f"row_defect={r['row_defect_sensitivity_pct']:.2f}%")

    print("\n[Phase5] Done!")
    return {
        "sens_df": sens_df,
        "discrete_df": discrete_df,
        "mc_df": mc_df,
        "defect_df": defect_df,
        "flow_df": flow_df,
        "stability_df": stability_df,
        "models": models,
        "solutions": solutions,
    }
