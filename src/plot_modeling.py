"""
Phase 2 绘图：模型对比、预测散点、残差、RSM-2 系数
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from src.config import (
    OUTPUT_FIGURES, OUTPUT_COLUMNS, OUTPUT_NAMES_CN, INPUT_COLUMNS, INPUT_NAMES_CN,
)
from src.utils import ensure_dir


def _ensure_fig():
    ensure_dir(OUTPUT_FIGURES)


# ────────────────────────────────────────────────────────────
#  fig_06: RMSE 对比柱状图
# ────────────────────────────────────────────────────────────
def plot_model_comparison_rmse(cv_df: pd.DataFrame):
    _ensure_fig()
    models_list = sorted(cv_df["model"].unique())
    n_models = len(models_list)
    n_targets = len(OUTPUT_COLUMNS)

    x = np.arange(n_models)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2c7bb6", "#d7191c", "#fdae61"]

    for j, target in enumerate(OUTPUT_COLUMNS):
        rmse_vals = []
        rmse_stds = []
        for m in models_list:
            sub = cv_df[(cv_df["model"] == m) & (cv_df["target"] == target)]
            rmse_vals.append(sub["RMSE_mean"].values[0])
            rmse_stds.append(sub["RMSE_std"].values[0])
        bars = ax.bar(x + j * width, rmse_vals, width, yerr=rmse_stds,
                     color=colors[j], capsize=3, label=OUTPUT_NAMES_CN[target])
        # 标注数值
        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                   f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models_list, fontsize=10)
    ax.set_ylabel("CV RMSE (mean +/- std)", fontsize=11)
    ax.set_title("Model Comparison: Repeated 5-Fold CV RMSE (20 repeats)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_06_model_comparison_rmse.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotModeling] Saved: fig_06_model_comparison_rmse.png")


# ────────────────────────────────────────────────────────────
#  fig_07-09: 预测 vs 真实散点图
# ────────────────────────────────────────────────────────────
def plot_pred_vs_true(trained_models: dict, X: np.ndarray, y_dict: dict):
    _ensure_fig()
    targets = OUTPUT_COLUMNS
    fnames = [
        "fig_07_pred_vs_true_thermal_resistance.png",
        "fig_08_pred_vs_true_pressure_drop.png",
        "fig_09_pred_vs_true_temperature_nonuniformity.png",
    ]

    for target, fname in zip(targets, fnames):
        _, pipe = trained_models[target]
        y_true = y_dict[target]
        y_pred = pipe.predict(X)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="k", linewidth=0.3, s=40,
                  color="#2c7bb6")

        # y=x 参考线
        vmin = min(y_true.min(), y_pred.min())
        vmax = max(y_true.max(), y_pred.max())
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
               "k--", linewidth=0.8, label="y=x")

        ax.set_xlabel(f"True {OUTPUT_NAMES_CN[target]}", fontsize=10)
        ax.set_ylabel(f"Predicted {OUTPUT_NAMES_CN[target]}", fontsize=10)
        ax.set_title(f"Predicted vs True: {OUTPUT_NAMES_CN[target]}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        path = os.path.join(OUTPUT_FIGURES, fname)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[PlotModeling] Saved: {fname}")


# ────────────────────────────────────────────────────────────
#  fig_10: 最佳模型残差图
# ────────────────────────────────────────────────────────────
def plot_residuals_best_model(trained_models: dict, X: np.ndarray, y_dict: dict):
    _ensure_fig()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for j, target in enumerate(OUTPUT_COLUMNS):
        ax = axes[j]
        _, pipe = trained_models[target]
        y_true = y_dict[target]
        y_pred = pipe.predict(X)
        residuals = y_true - y_pred

        ax.scatter(y_pred, residuals, alpha=0.7, edgecolor="k", linewidth=0.3,
                  s=30, color="#2c7bb6")
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"Predicted", fontsize=9)
        ax.set_ylabel("Residual", fontsize=9)
        ax.set_title(f"{OUTPUT_NAMES_CN[target]}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

        # 标注 +/-2σ 带
        std_res = np.std(residuals)
        ax.axhline(y=2*std_res, color="red", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.axhline(y=-2*std_res, color="red", linestyle=":", linewidth=0.6, alpha=0.5)

    fig.suptitle("Residuals of Best Surrogate Models", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_10_residuals_best_model.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotModeling] Saved: fig_10_residuals_best_model.png")


# ────────────────────────────────────────────────────────────
#  fig_11: RSM-2 标准化系数重要性
# ────────────────────────────────────────────────────────────
def plot_rsm2_coefficient_importance(coef_df: pd.DataFrame):
    _ensure_fig()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for j, target in enumerate(OUTPUT_COLUMNS):
        ax = axes[j]
        sub = coef_df[coef_df["target"] == target].copy()
        # 按绝对值排序，取前 10
        sub = sub.sort_values("abs_coefficient_std", ascending=False).head(10)
        # 反转使最大在顶
        sub = sub.iloc[::-1]

        colors = ["#d7191c" if s == "-" else "#2c7bb6" for s in sub["sign"]]
        ax.barh(range(len(sub)), sub["abs_coefficient_std"], color=colors, edgecolor="k", linewidth=0.3)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["term"], fontsize=7)
        ax.set_xlabel("|Standardized Coefficient|", fontsize=9)
        ax.set_title(f"{OUTPUT_NAMES_CN[target]}", fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2c7bb6", label="Positive"),
            Patch(facecolor="#d7191c", label="Negative"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

    fig.suptitle("RSM-2 Standardized Coefficient Importance (Top 10 terms)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_11_rsm2_coefficient_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotModeling] Saved: fig_11_rsm2_coefficient_importance.png")
