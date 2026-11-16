"""
Phase 3 绘图：Pareto 前沿、综合评分、雷达图
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from src.config import (
    OUTPUT_FIGURES, OUTPUT_COLUMNS, OUTPUT_NAMES_CN, INPUT_COLUMNS,
)
from src.utils import ensure_dir


def _ensure_fig():
    ensure_dir(OUTPUT_FIGURES)


# ────────────────────────────────────────────────────────────
#  fig_12-14: Pareto 2D pairwise projections
# ────────────────────────────────────────────────────────────
def plot_pareto_2d_projections(
    Y_all: np.ndarray,
    pareto_mask: np.ndarray,
    Y_pareto: np.ndarray,
    key_samples: dict,
):
    """
    key_samples: {label: (y1, y2, y3)}
    """
    _ensure_fig()

    pairs = [
        (0, 1, "fig_12_pareto_y1_y2.png", OUTPUT_NAMES_CN[OUTPUT_COLUMNS[0]],
         OUTPUT_NAMES_CN[OUTPUT_COLUMNS[1]]),
        (0, 2, "fig_13_pareto_y1_y3.png", OUTPUT_NAMES_CN[OUTPUT_COLUMNS[0]],
         OUTPUT_NAMES_CN[OUTPUT_COLUMNS[2]]),
        (1, 2, "fig_14_pareto_y2_y3.png", OUTPUT_NAMES_CN[OUTPUT_COLUMNS[1]],
         OUTPUT_NAMES_CN[OUTPUT_COLUMNS[2]]),
    ]

    for ix, iy, fname, xlabel, ylabel in pairs:
        fig, ax = plt.subplots(figsize=(6.5, 5))

        # All points (gray)
        ax.scatter(Y_all[:, ix], Y_all[:, iy], alpha=0.3, s=15, color="gray",
                  label="All Candidates", zorder=1)

        # Pareto (highlighted)
        ax.scatter(Y_pareto[:, ix], Y_pareto[:, iy], alpha=0.9, s=30, color="#d7191c",
                  edgecolors="k", linewidth=0.3, label="Pareto Front", zorder=2)

        # Key samples
        colors_map = {"#64": "#2c7bb6", "#48": "#fdae61", "#13": "#7b3294", "#80": "#008837"}
        for label, (y1, y2, y3) in key_samples.items():
            y_vals = [y1, y2, y3]
            color = colors_map.get(label, "black")
            ax.scatter(y_vals[ix], y_vals[iy], s=80, color=color, edgecolors="k",
                      linewidth=0.8, zorder=3)
            ax.annotate(label, (y_vals[ix], y_vals[iy]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8,
                       fontweight="bold")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Pareto Front: {xlabel} vs {ylabel}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        path = os.path.join(OUTPUT_FIGURES, fname)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[PlotOpt] Saved: {fname}")


# ────────────────────────────────────────────────────────────
#  fig_15: 3D Pareto front
# ────────────────────────────────────────────────────────────
def plot_pareto_3d(
    Y_all: np.ndarray,
    pareto_mask: np.ndarray,
    Y_pareto: np.ndarray,
    key_samples: dict,
):
    _ensure_fig()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # All candidates
    ax.scatter(Y_all[:, 0], Y_all[:, 1], Y_all[:, 2],
              alpha=0.15, s=10, color="gray", label="All Candidates")

    # Pareto
    ax.scatter(Y_pareto[:, 0], Y_pareto[:, 1], Y_pareto[:, 2],
              alpha=0.9, s=35, color="#d7191c", edgecolors="k", linewidth=0.3,
              label="Pareto Front")

    # Key samples
    colors_map = {"#64": "#2c7bb6", "#48": "#fdae61", "#13": "#7b3294", "#80": "#008837"}
    for label, (y1, y2, y3) in key_samples.items():
        color = colors_map.get(label, "black")
        ax.scatter(y1, y2, y3, s=100, color=color, edgecolors="k", linewidth=1, zorder=5)
        ax.text(y1, y2, y3, f" {label}", fontsize=8, fontweight="bold")

    ax.set_xlabel(OUTPUT_NAMES_CN[OUTPUT_COLUMNS[0]], fontsize=9)
    ax.set_ylabel(OUTPUT_NAMES_CN[OUTPUT_COLUMNS[1]], fontsize=9)
    ax.set_zlabel(OUTPUT_NAMES_CN[OUTPUT_COLUMNS[2]], fontsize=9)
    ax.set_title("3D Pareto Front", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_15_3d_pareto_front.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotOpt] Saved: fig_15_3d_pareto_front.png")


# ────────────────────────────────────────────────────────────
#  fig_16-17: Weighted Sum / TOPSIS Top 10 bar charts
# ────────────────────────────────────────────────────────────
def plot_top10_barcharts(top10_df: pd.DataFrame, score_col: str, title: str, fname: str):
    """
    top10_df 需包含：label, score
    """
    _ensure_fig()

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = top10_df["label"].values
    scores = top10_df[score_col].values

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))

    bars = ax.barh(range(len(labels)), scores, color=colors, edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(score_col, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # 标注数值
    for bar, val in zip(bars, scores):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
               f"{val:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotOpt] Saved: {fname}")


def plot_sample_space_top10(sample_results: dict):
    """样本空间加权和 + TOPSIS Top 10"""
    res_df = sample_results["result"]
    ws_rank = sample_results["ws_rank"]
    topsis_rank = sample_results["topsis_rank"]

    # Weighted Sum Top 10
    top10_ws = res_df.iloc[ws_rank[:10]]
    ws_data = pd.DataFrame({
        "label": [f"#{int(sid)}" for sid in top10_ws["sample_id"]],
        "Weighted Sum (lower=better)": top10_ws["weighted_sum"].values,
    })
    plot_top10_barcharts(ws_data, "Weighted Sum (lower=better)",
                         "Top 10 by Equal-Weight Weighted Sum (Sample Space)",
                         "fig_16_weighted_sum_top10.png")

    # TOPSIS Top 10
    top10_topsis = res_df.iloc[topsis_rank[:10]]
    topsis_data = pd.DataFrame({
        "label": [f"#{int(sid)}" for sid in top10_topsis["sample_id"]],
        "TOPSIS Score (higher=better)": top10_topsis["topsis_score"].values,
    })
    plot_top10_barcharts(topsis_data, "TOPSIS Score (higher=better)",
                         "Top 10 by TOPSIS (Sample Space)",
                         "fig_17_topsis_top10.png")


# ────────────────────────────────────────────────────────────
#  fig_18: Grouped bar chart comparing final candidates
# ────────────────────────────────────────────────────────────
def plot_final_candidates_comparison(comp_df: pd.DataFrame):
    """
    Grouped bar chart of normalized scores for final candidates.
    """
    _ensure_fig()

    # Compute normalized scores for the comparison table
    candidates = comp_df.copy()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    n_cand = len(candidates)
    x = np.arange(n_cand)
    width = 0.22

    targets = OUTPUT_COLUMNS
    colors = ["#2c7bb6", "#d7191c", "#fdae61"]

    # Min-max normalize across candidates for display
    true_cols = ["true_y1", "true_y2", "true_y3"]
    for j, target in enumerate(targets):
        vals = candidates[true_cols[j]].values
        vmin, vmax = vals.min(), vals.max()
        denom = vmax - vmin if vmax > vmin else 1.0
        norm_vals = (vals - vmin) / denom

        bars = ax.bar(x + j * width, norm_vals, width, color=colors[j],
                     label=OUTPUT_NAMES_CN[target], edgecolor="k", linewidth=0.3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(candidates["note"].values, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Normalized Score (lower = better)", fontsize=10)
    ax.set_title("Comparison of Final Candidate Designs", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_18_final_candidates_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotOpt] Saved: fig_18_final_candidates_comparison.png")
