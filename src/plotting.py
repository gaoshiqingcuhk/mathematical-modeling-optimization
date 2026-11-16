"""
绘图函数
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from src.config import (
    OUTPUT_FIGURES, INPUT_COLUMNS, OUTPUT_COLUMNS,
    INPUT_NAMES_CN, OUTPUT_NAMES_CN, X_BOUNDS,
)
from src.utils import ensure_dir


def _ensure_fig_dir():
    ensure_dir(OUTPUT_FIGURES)


def _setup_ax(ax, title, xlabel, ylabel):
    """公共样式设置"""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_main_effects(df: pd.DataFrame):
    """
    生成三个主效应图：
    - fig_02: 各参数对无量纲热阻的效应
    - fig_03: 各参数对无量纲压降的效应
    - fig_04: 各参数对无量纲温度非均匀性的效应
    """
    _ensure_fig_dir()

    # 参数分组
    # x1 = pin_width_ratio 的取值
    x1_vals = sorted(df["pin_width_ratio"].unique())  # [0, 0.1, 0.2, 0.3]
    # x2 = manifold_depth_height_ratio 的取值
    x2_vals = sorted(df["manifold_depth_height_ratio"].unique())  # [3.0, 3.5, 4.0, 4.5]
    # x3 = pin_rows 的取值
    x3_vals = sorted(df["pin_rows"].unique())  # [0, 2, 4, 6, 8, 10]

    # 三个输入变量的列表
    input_info = [
        ("pin_width_ratio", x1_vals, "针肋宽度比 $x_1$"),
        ("manifold_depth_height_ratio", x2_vals, "歧管深高比 $x_2$"),
        ("pin_rows", x3_vals, "针肋排数 $x_3$"),
    ]

    # 三个输出变量映射
    output_info = [
        ("thermal_resistance", "无量纲热阻", "fig_02_main_effects_heat_resistance.png"),
        ("pressure_drop", "无量纲压降", "fig_03_main_effects_pressure_drop.png"),
        ("temperature_nonuniformity", "无量纲温度非均匀性", "fig_04_main_effects_temperature_nonuniformity.png"),
    ]

    for out_col, out_name_cn, fname in output_info:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(f"结构参数对{out_name_cn}的影响", fontsize=13, fontweight="bold", y=1.02)

        for ax_idx, (in_col, vals, label) in enumerate(input_info):
            ax = axes[ax_idx]
            # 计算每个 x_i 水平下 y_j 的均值
            means = df.groupby(in_col)[out_col].mean()
            # 同时计算 std
            stds = df.groupby(in_col)[out_col].std()

            x_plot = means.index.values
            y_plot = means.values
            y_err = stds.values

            ax.errorbar(x_plot, y_plot, yerr=y_err, marker='o', capsize=4,
                       linestyle='-', linewidth=1.5, markersize=6, color='#2c7bb6')

            _setup_ax(ax,
                     title=f"{INPUT_NAMES_CN[in_col]}",
                     xlabel=label,
                     ylabel=f"{out_name_cn} (均值 ± 1σ)")

        plt.tight_layout()
        path = os.path.join(OUTPUT_FIGURES, fname)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plotting] 已保存: {fname}")


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    fig_05: 变量间相关性热力图
    """
    _ensure_fig_dir()

    all_cols = INPUT_COLUMNS + OUTPUT_COLUMNS
    corr = df[all_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # 标注
    labels = [INPUT_NAMES_CN.get(c, OUTPUT_NAMES_CN.get(c, c)) for c in all_cols]
    ax.set_xticks(range(len(all_cols)))
    ax.set_yticks(range(len(all_cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # 在格内写数值
    for i in range(len(all_cols)):
        for j in range(len(all_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.3f}", ha="center", va="center",
                   fontsize=8, color="black")

    ax.set_title("变量相关性热力图", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_05_correlation_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plotting] 已保存: fig_05_correlation_heatmap.png")
