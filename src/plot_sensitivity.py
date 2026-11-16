"""
Phase 5 绘图：加工误差敏感性、Monte Carlo、工况扰动
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from src.config import OUTPUT_FIGURES, OUTPUT_COLUMNS, OUTPUT_NAMES_CN
from src.utils import ensure_dir


def _ensure_fig():
    ensure_dir(OUTPUT_FIGURES)


# ══════════════════════════════════════════════════════════════
#  fig_25: Local sensitivity heatmap
# ══════════════════════════════════════════════════════════════
def plot_local_sensitivity_heatmap(sens_df):
    """
    Heatmap: solutions × (parameter, perturbation) showing max |S| across targets.
    """
    _ensure_fig()

    solutions = sorted(sens_df["solution"].unique())
    params = sorted(sens_df["perturbed_parameter"].unique())
    levels = sorted(sens_df["perturbation_level"].unique(), key=lambda x: int(x.replace("%", "")))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for pi, param in enumerate(params):
        ax = axes[pi]
        # Build matrix: solutions × levels, value = max |S| across 3 targets
        matrix = np.zeros((len(solutions), len(levels)))
        for si, sol in enumerate(solutions):
            for li, lvl in enumerate(levels):
                sub = sens_df[(sens_df["solution"] == sol) &
                             (sens_df["perturbed_parameter"] == param) &
                             (sens_df["perturbation_level"] == lvl)]
                if len(sub) > 0:
                    matrix[si, li] = sub["relative_sensitivity"].abs().max()

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, fontsize=9)
        ax.set_yticks(range(len(solutions)))
        ax.set_yticklabels([f"#{s}" for s in solutions], fontsize=9)
        ax.set_title(f"Sensitivity to {param.replace('_',' ').title()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Perturbation Level", fontsize=10)

        # Annotate cells
        for si in range(len(solutions)):
            for li in range(len(levels)):
                val = matrix[si, li]
                ax.text(li, si, f"{val:.3f}", ha="center", va="center", fontsize=8,
                       color="white" if val > 0.5 else "black")

        plt.colorbar(im, ax=ax, shrink=0.8, label="Max Absolute Sensitivity |S|")

    fig.suptitle("Local Sensitivity of Design Parameters (Absolute Relative Sensitivity |S|)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_25_local_sensitivity_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotSens] Saved: fig_25_local_sensitivity_heatmap.png")


# ══════════════════════════════════════════════════════════════
#  fig_26: Monte Carlo composite score distribution #64 vs #48
# ══════════════════════════════════════════════════════════════
def plot_monte_carlo_distribution(models, solutions_info):
    """
    Generate MC distribution for #64 and #48 composite scores.
    Re-runs MC with just these two solutions.
    """
    _ensure_fig()

    from src.sensitivity import predict_all

    rng = np.random.default_rng(2026)
    n = 10000
    sol_data = {}

    for sol_name, x1_nom, x2_nom, x3_nom in solutions_info:
        if sol_name not in ("64", "48"):
            continue

        x1_s = rng.normal(x1_nom, 0.03 * x1_nom, n)
        x2_s = rng.normal(x2_nom, 0.03 * x2_nom, n)
        x1_s = np.clip(x1_s, 0.10, 0.30)
        x2_s = np.clip(x2_s, 3.0, 4.5)

        X_p = np.column_stack([x1_s, x2_s, np.full(n, x3_nom)])
        Y_p = predict_all(models, X_p)

        x_nom = np.array([x1_nom, x2_nom, x3_nom])
        y_nom = predict_all(models, x_nom.reshape(1, -1))[0]

        Y_all = np.vstack([y_nom.reshape(1, -1), Y_p])
        y_min, y_max = Y_all.min(axis=0), Y_all.max(axis=0)
        denom = y_max - y_min
        denom[denom == 0] = 1e-12
        Z_all = (Y_all - y_min) / denom
        w = np.array([1/3, 1/3, 1/3])
        ws = Z_all @ w  # (n+1,)
        ws_nom = ws[0]
        ws_mc = ws[1:]

        sol_data[sol_name] = {"ws_nom": ws_nom, "ws_mc": ws_mc, "y_nom": y_nom, "Y_p": Y_p}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram overlay
    ax = axes[0]
    colors = {"64": "#2c7bb6", "48": "#fdae61"}
    for sol_name in ("64", "48"):
        data = sol_data[sol_name]
        ax.hist(data["ws_mc"], bins=60, alpha=0.5, color=colors[sol_name],
               label=f"#{sol_name} (nom={data['ws_nom']:.3f}, std={np.std(data['ws_mc']):.4f})",
               edgecolor="k", linewidth=0.3)
        ax.axvline(data["ws_nom"], color=colors[sol_name], linestyle="--", linewidth=2)

    ax.set_xlabel("Composite Weighted Sum Score (lower=better)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("MC Manufacturing Error: Composite Score Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: box plot
    ax = axes[1]
    bp_data = [sol_data["64"]["ws_mc"], sol_data["48"]["ws_mc"]]
    bp = ax.boxplot(bp_data, labels=["#64", "#48"], patch_artist=True, widths=0.4)
    bp["boxes"][0].set_facecolor("#2c7bb6")
    bp["boxes"][1].set_facecolor("#fdae61")

    # Mark nominal
    ax.scatter([1, 2], [sol_data["64"]["ws_nom"], sol_data["48"]["ws_nom"]],
              marker="*", color="red", s=150, zorder=5, label="Nominal")
    ax.set_ylabel("Composite WS Score (lower=better)", fontsize=10)
    ax.set_title("MC Score Distribution: #64 vs #48", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Manufacturing Error Impact: #64 vs #48 (10,000 MC Samples, 3% CV)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_26_monte_carlo_distribution_64_vs_48.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotSens] Saved: fig_26_monte_carlo_distribution_64_vs_48.png")

    return sol_data


# ══════════════════════════════════════════════════════════════
#  fig_27: Monte Carlo boxplot for individual targets
# ══════════════════════════════════════════════════════════════
def plot_monte_carlo_boxplot_targets(sol_data):
    """
    Box plot of y1, y2, y3 for #64 and #48 under MC perturbations.
    """
    _ensure_fig()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    target_names = ["Thermal Resistance", "Pressure Drop", "Temperature Nonuniformity"]
    colors = {"64": "#2c7bb6", "48": "#fdae61"}

    for j, (target, tname) in enumerate(zip(OUTPUT_COLUMNS, target_names)):
        ax = axes[j]
        positions = [1, 2]
        bp_data = []
        for sol_name in ("64", "48"):
            bp_data.append(sol_data[sol_name]["Y_p"][:, j])

        bp = ax.boxplot(bp_data, labels=["#64", "#48"], patch_artist=True, widths=0.4)
        for k, sol_name in enumerate(("64", "48")):
            bp["boxes"][k].set_facecolor(colors[sol_name])
            nom = sol_data[sol_name]["y_nom"][j]
            ax.scatter(k + 1, nom, marker="*", color="red", s=100, zorder=5)

        ax.set_ylabel(tname, fontsize=9)
        ax.set_title(f"{tname}", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Target Manufacturing Error Sensitivity: #64 vs #48", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_27_monte_carlo_boxplot_targets.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotSens] Saved: fig_27_monte_carlo_boxplot_targets.png")


# ══════════════════════════════════════════════════════════════
#  fig_28: Pin rows discrete sensitivity
# ══════════════════════════════════════════════════════════════
def plot_pin_rows_discrete_sensitivity(discrete_df):
    """
    Grouped bar chart showing % change in y1/y2/y3 when x3 changes.
    """
    _ensure_fig()

    solutions = discrete_df["solution"].unique()
    n_sol = len(solutions)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    target_names = ["Thermal Resistance", "Pressure Drop", "Temperature Nonuniformity"]
    delta_cols = ["delta_y1_percent", "delta_y2_percent", "delta_y3_percent"]
    colors = {"64": "#2c7bb6", "48": "#fdae61", "14": "#d7191c", "80": "#008837"}

    for j, (dcol, tname) in enumerate(zip(delta_cols, target_names)):
        ax = axes[j]
        bar_positions = []
        bar_values = []
        bar_colors = []
        bar_labels = []

        pos = 0
        for sol in solutions:
            sub = discrete_df[discrete_df["solution"] == sol]
            for _, r in sub.iterrows():
                bar_positions.append(pos)
                bar_values.append(r[dcol])
                bar_colors.append(colors.get(sol, "gray"))
                bar_labels.append(f"#{sol}\nx3={int(r['test_x3'])}")
                pos += 1
            pos += 1  # gap between solutions

        bars = ax.bar(bar_positions, bar_values, color=bar_colors, edgecolor="k", linewidth=0.5)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(bar_labels, fontsize=6, rotation=45)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("% Change", fontsize=9)
        ax.set_title(tname, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Color for reference lines
        for i, bar in enumerate(bars):
            val = bar_values[i]
            color = "darkgreen" if abs(val) < 2 else ("darkred" if abs(val) > 5 else "orange")
            ax.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + (0.3 if val >= 0 else -0.8),
                   f"{val:.1f}%", ha="center", fontsize=6.5, color=color)

    fig.suptitle("Discrete Pin Rows Sensitivity: Effect of Changing x3 on Design Performance",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_28_pin_rows_discrete_sensitivity.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotSens] Saved: fig_28_pin_rows_discrete_sensitivity.png")


# ══════════════════════════════════════════════════════════════
#  fig_29: Flow rate fluctuation
# ══════════════════════════════════════════════════════════════
def plot_flow_rate_fluctuation(flow_df):
    """
    Line plot showing how y1, y2, y3 change with flow rate for #64 and #48.
    """
    _ensure_fig()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    target_names = ["Thermal Resistance", "Pressure Drop", "Temperature Nonuniformity"]
    y_cols = ["y1_scaled", "y2_scaled", "y3_scaled"]
    colors = {"64": "#2c7bb6", "48": "#fdae61"}

    for j, (ycol, tname) in enumerate(zip(y_cols, target_names)):
        ax = axes[j]
        for sol in ("64", "48"):
            sub = flow_df[flow_df["solution"] == sol]
            mr = sub["m_dot_ratio"].values
            yv = sub[ycol].values
            ax.plot(mr, yv, marker="o", color=colors[sol], linewidth=2, markersize=8, label=f"#{sol}")

        ax.set_xlabel("Mass Flow Rate Ratio", fontsize=9)
        ax.set_ylabel(tname, fontsize=9)
        ax.set_title(tname, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Flow Rate Fluctuation: Engineering Approximation (#64 vs #48)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_29_flow_rate_fluctuation.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotSens] Saved: fig_29_flow_rate_fluctuation.png")


# ══════════════════════════════════════════════════════════════
#  Master plot function
# ══════════════════════════════════════════════════════════════
def plot_all_phase5(p5_results):
    """Generate all Phase 5 figures."""
    print("\n[PlotSens] Generating Phase 5 figures...")

    sens_df = p5_results["sens_df"]
    models = p5_results["models"]
    solutions = p5_results["solutions"]
    discrete_df = p5_results["discrete_df"]
    flow_df = p5_results["flow_df"]

    plot_local_sensitivity_heatmap(sens_df)
    sol_data = plot_monte_carlo_distribution(models, solutions)
    plot_monte_carlo_boxplot_targets(sol_data)
    plot_pin_rows_discrete_sensitivity(discrete_df)
    plot_flow_rate_fluctuation(flow_df)

    print("[PlotSens] All Phase 5 figures generated.")
