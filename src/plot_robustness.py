"""
Phase 4 绘图：权重敏感性、鲁棒性对比
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
#  fig_19: Scenario best solutions comparison
# ══════════════════════════════════════════════════════════════
def plot_scenario_best_solutions(scenario_df, analysis_result):
    """
    Grouped bar chart: for each scenario, show the best WS and TOPSIS solutions.
    """
    _ensure_fig()

    scenarios = scenario_df["scenario"].values
    descriptions = scenario_df["description"].values
    n = len(scenarios)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: Best WS ──
    ax = axes[0]
    x = np.arange(n)
    width = 0.35

    unique_labels = []
    label_to_color = {}
    color_cycle = plt.cm.tab10(np.linspace(0, 1, 10))

    for i in range(n):
        lbl = scenario_df.iloc[i]["best_ws_candidate"]
        if lbl not in label_to_color:
            label_to_color[lbl] = color_cycle[len(label_to_color) % 10]
            unique_labels.append(lbl)

    bars = []
    for i in range(n):
        lbl = scenario_df.iloc[i]["best_ws_candidate"]
        bar = ax.bar(i, 1, width, color=label_to_color[lbl], edgecolor="k", linewidth=0.5)
        bars.append(bar)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=7, rotation=30)
    ax.set_ylabel("Best Candidate (WS)", fontsize=10)
    ax.set_title("Best Weighted-Sum Solution per Scenario", fontsize=11, fontweight="bold")

    # Annotate bars with candidate ID
    for i in range(n):
        lbl = scenario_df.iloc[i]["best_ws_candidate"]
        ax.text(i, 0.5, f"#{lbl}", ha="center", va="center", fontsize=8, fontweight="bold")

    # Custom legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=f"#{lbl}") for lbl, c in label_to_color.items()]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right")

    # ── Right: Best TOPSIS ──
    ax = axes[1]
    unique_labels_t = []
    label_to_color_t = {}
    for i in range(n):
        lbl = scenario_df.iloc[i]["best_topsis_candidate"]
        if lbl not in label_to_color_t:
            label_to_color_t[lbl] = color_cycle[len(label_to_color_t) % 10]
            unique_labels_t.append(lbl)

    for i in range(n):
        lbl = scenario_df.iloc[i]["best_topsis_candidate"]
        ax.bar(i, 1, width, color=label_to_color_t[lbl], edgecolor="k", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=7, rotation=30)
    ax.set_ylabel("Best Candidate (TOPSIS)", fontsize=10)
    ax.set_title("Best TOPSIS Solution per Scenario", fontsize=11, fontweight="bold")

    for i in range(n):
        lbl = scenario_df.iloc[i]["best_topsis_candidate"]
        ax.text(i, 0.5, f"#{lbl}", ha="center", va="center", fontsize=8, fontweight="bold")

    legend_patches_t = [Patch(color=c, label=f"#{lbl}") for lbl, c in label_to_color_t.items()]
    ax.legend(handles=legend_patches_t, fontsize=7, loc="upper right")

    fig.suptitle("Optimal Solutions Under Different Weight Scenarios", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_19_weight_scenario_best_solutions.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotRobust] Saved: fig_19_weight_scenario_best_solutions.png")


# ══════════════════════════════════════════════════════════════
#  fig_20: Weight sampling best frequency
# ══════════════════════════════════════════════════════════════
def plot_weight_sampling_best_frequency(freq_table, top_n=15):
    """
    Horizontal bar chart showing top candidates by best frequency.
    """
    _ensure_fig()

    df = freq_table.head(top_n).copy()
    labels = [f"#{c}" for c in df["candidate"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # ── WS best frequency ──
    ax = axes[0]
    freqs_ws = df["best_freq_ws"].values * 100  # percentage
    colors_ws = ["#d7191c" if lbl == "#64" else ("#fdae61" if lbl == "#48" else "#5a9bd5")
                 for lbl in labels]

    bars = ax.barh(range(len(labels)), freqs_ws, color=colors_ws, edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Best Frequency (%)", fontsize=10)
    ax.set_title("Weighted Sum: Best Frequency", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, freqs_ws):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=7)

    # ── TOPSIS best frequency ──
    ax = axes[1]
    freqs_t = df["best_freq_topsis"].values * 100
    colors_t = ["#d7191c" if lbl == "#64" else ("#fdae61" if lbl == "#48" else "#5a9bd5")
                for lbl in labels]

    bars = ax.barh(range(len(labels)), freqs_t, color=colors_t, edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Best Frequency (%)", fontsize=10)
    ax.set_title("TOPSIS: Best Frequency", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, freqs_t):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=7)

    fig.suptitle("Candidate Optimality Frequency Under Random Weight Sampling (10000 Samples)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_20_weight_sampling_best_frequency.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotRobust] Saved: fig_20_weight_sampling_best_frequency.png")


# ══════════════════════════════════════════════════════════════
#  fig_21: Weight simplex regions (2D projection)
# ══════════════════════════════════════════════════════════════
def plot_weight_simplex_regions(analysis_result, top_n=8):
    """
    Scatter plot in w1-w2 plane showing which candidate is optimal
    for each weight sample. w3 = 1 - w1 - w2.
    """
    _ensure_fig()

    best_ws_idx = analysis_result["best_ws_idx"]  # (k,)
    weights = analysis_result["weights"]  # (k, 3)
    labels = analysis_result["candidate_labels"]
    k = weights.shape[0]

    # Find top candidates by frequency
    from collections import Counter
    freq = Counter(best_ws_idx)
    top_candidates = [idx for idx, _ in freq.most_common(top_n)]

    # For display, subsample if too many points
    if k > 5000:
        rng = np.random.default_rng(42)
        plot_idx = rng.choice(k, 5000, replace=False)
    else:
        plot_idx = np.arange(k)

    w1_plot = weights[plot_idx, 0]
    w2_plot = weights[plot_idx, 1]
    best_plot = best_ws_idx[plot_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── WS Regions ──
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_candidates)))
    color_map = {idx: colors[i] for i, idx in enumerate(top_candidates)}

    for i, cand_idx in enumerate(top_candidates):
        mask = best_plot == cand_idx
        if mask.sum() > 0:
            lbl = labels[cand_idx]
            ax.scatter(w1_plot[mask], w2_plot[mask], s=8, alpha=0.6,
                      color=color_map[cand_idx], label=f"#{lbl}", edgecolors="none")

    # Draw simplex boundary
    t = np.linspace(0, 1, 100)
    ax.plot(t, 1 - t, "k--", linewidth=0.8, alpha=0.5)
    ax.plot(np.zeros(100), t, "k--", linewidth=0.8, alpha=0.5)
    ax.plot(t, np.zeros(100), "k--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("w1 (Thermal Resistance)", fontsize=10)
    ax.set_ylabel("w2 (Pressure Drop)", fontsize=10)
    ax.set_title("WS Optimal Regions in Weight Space", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # ── TOPSIS Regions ──
    ax = axes[1]
    best_topsis_idx = analysis_result["best_topsis_idx"]
    best_t_plot = best_topsis_idx[plot_idx]

    freq_t = Counter(best_topsis_idx)
    top_candidates_t = [idx for idx, _ in freq_t.most_common(top_n)]
    colors_t = plt.cm.tab10(np.linspace(0, 1, len(top_candidates_t)))
    color_map_t = {idx: colors_t[i] for i, idx in enumerate(top_candidates_t)}

    for i, cand_idx in enumerate(top_candidates_t):
        mask = best_t_plot == cand_idx
        if mask.sum() > 0:
            lbl = labels[cand_idx]
            ax.scatter(w1_plot[mask], w2_plot[mask], s=8, alpha=0.6,
                      color=color_map_t[cand_idx], label=f"#{lbl}", edgecolors="none")

    ax.plot(t, 1 - t, "k--", linewidth=0.8, alpha=0.5)
    ax.plot(np.zeros(100), t, "k--", linewidth=0.8, alpha=0.5)
    ax.plot(t, np.zeros(100), "k--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("w1 (Thermal Resistance)", fontsize=10)
    ax.set_ylabel("w2 (Pressure Drop)", fontsize=10)
    ax.set_title("TOPSIS Optimal Regions in Weight Space", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    fig.suptitle("Weight Space Partition: Which Design is Optimal?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_21_weight_simplex_regions.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotRobust] Saved: fig_21_weight_simplex_regions.png")


# ══════════════════════════════════════════════════════════════
#  fig_22: Rank distribution #64 vs #48
# ══════════════════════════════════════════════════════════════
def plot_rank_distribution_64_vs_48(analysis_result):
    """
    Box plot comparing rank distributions of #64 and #48.
    """
    _ensure_fig()

    labels = analysis_result["candidate_labels"]
    idx_64 = np.where(labels == "64")[0]
    idx_48 = np.where(labels == "48")[0]

    if len(idx_64) == 0 or len(idx_48) == 0:
        print("[WARN] #64 or #48 not found, skipping fig_22")
        return

    idx_64 = idx_64[0]
    idx_48 = idx_48[0]

    ws_rank_64 = analysis_result["ws_ranks"][idx_64, :]
    ws_rank_48 = analysis_result["ws_ranks"][idx_48, :]
    topsis_rank_64 = analysis_result["topsis_ranks"][idx_64, :]
    topsis_rank_48 = analysis_result["topsis_ranks"][idx_48, :]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ── WS Ranks ──
    ax = axes[0]
    bp1 = ax.boxplot([ws_rank_64, ws_rank_48], labels=["#64", "#48"],
                     patch_artist=True, widths=0.5)
    bp1["boxes"][0].set_facecolor("#2c7bb6")
    bp1["boxes"][1].set_facecolor("#fdae61")

    # Add mean markers
    ax.scatter([1, 2], [ws_rank_64.mean(), ws_rank_48.mean()],
              marker="D", color="red", s=50, zorder=5, label="Mean")

    ax.set_ylabel("Weighted Sum Rank (1=best)", fontsize=10)
    ax.set_title("WS Rank Distribution: #64 vs #48", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ── TOPSIS Ranks ──
    ax = axes[1]
    bp2 = ax.boxplot([topsis_rank_64, topsis_rank_48], labels=["#64", "#48"],
                     patch_artist=True, widths=0.5)
    bp2["boxes"][0].set_facecolor("#2c7bb6")
    bp2["boxes"][1].set_facecolor("#fdae61")

    ax.scatter([1, 2], [topsis_rank_64.mean(), topsis_rank_48.mean()],
              marker="D", color="red", s=50, zorder=5, label="Mean")

    ax.set_ylabel("TOPSIS Rank (1=best)", fontsize=10)
    ax.set_title("TOPSIS Rank Distribution: #64 vs #48", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Rank Stability Under 10,000 Random Weight Samplings",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_22_rank_distribution_64_vs_48.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotRobust] Saved: fig_22_rank_distribution_64_vs_48.png")


# ══════════════════════════════════════════════════════════════
#  fig_23: Score distribution #64 vs #48
# ══════════════════════════════════════════════════════════════
def plot_score_distribution_64_vs_48(analysis_result):
    """
    Overlaid histograms of WS scores for #64 and #48.
    """
    _ensure_fig()

    labels = analysis_result["candidate_labels"]
    idx_64 = np.where(labels == "64")[0]
    idx_48 = np.where(labels == "48")[0]

    if len(idx_64) == 0 or len(idx_48) == 0:
        print("[WARN] #64 or #48 not found, skipping fig_23")
        return

    idx_64 = idx_64[0]
    idx_48 = idx_48[0]

    ws_64 = analysis_result["ws_scores"][idx_64, :]
    ws_48 = analysis_result["ws_scores"][idx_48, :]

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = 60
    ax.hist(ws_64, bins=bins, alpha=0.6, color="#2c7bb6", label=f"#64 (mean={ws_64.mean():.4f}, std={ws_64.std():.4f})",
            edgecolor="k", linewidth=0.3)
    ax.hist(ws_48, bins=bins, alpha=0.6, color="#fdae61", label=f"#48 (mean={ws_48.mean():.4f}, std={ws_48.std():.4f})",
            edgecolor="k", linewidth=0.3)

    # CVaR95 markers
    n_worst = max(1, int(len(ws_64) * 0.05))
    cvar64 = np.mean(np.sort(ws_64)[-n_worst:])
    cvar48 = np.mean(np.sort(ws_48)[-n_worst:])
    ax.axvline(cvar64, color="#2c7bb6", linestyle="--", linewidth=1.5, alpha=0.8,
              label=f"#64 CVaR95={cvar64:.4f}")
    ax.axvline(cvar48, color="#fdae61", linestyle="--", linewidth=1.5, alpha=0.8,
              label=f"#48 CVaR95={cvar48:.4f}")

    ax.set_xlabel("Weighted Sum Score (lower = better)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("WS Score Distribution Under 10,000 Random Weights: #64 vs #48",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_23_score_distribution_64_vs_48.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotRobust] Saved: fig_23_score_distribution_64_vs_48.png")


# ══════════════════════════════════════════════════════════════
#  fig_24: Robustness metrics comparison
# ══════════════════════════════════════════════════════════════
def plot_robustness_metrics_comparison(metrics_df, analysis_result):
    """
    Radar-like grouped bar chart comparing #64, #48, and the robust optimum
    across key metrics: mean_score, std_score, cvar95, best_freq.
    """
    _ensure_fig()

    # Find #64, #48, and the top robust candidate
    candidates_to_plot = []
    for cid in ["64", "48"]:
        row = metrics_df[metrics_df["candidate"] == cid]
        if len(row) > 0:
            candidates_to_plot.append(row.iloc[0])

    # Also add the #1 robust candidate (if not already included)
    top_robust = metrics_df.iloc[0]
    if top_robust["candidate"] not in ["64", "48"]:
        candidates_to_plot.append(top_robust)

    if len(candidates_to_plot) < 2:
        print("[WARN] Not enough candidates for fig_24")
        return

    n_candidates = len(candidates_to_plot)
    labels = [f"#{c['candidate']}" for c in candidates_to_plot]

    # Metrics to compare (all lower-is-better except best_freq)
    metric_names = ["Mean Score", "Std Score", "CVaR95", "Best Freq (WS)"]
    # For best_freq, we use (1 - best_freq) to make it "lower is better" for display
    metric_values = []
    for c in candidates_to_plot:
        metric_values.append([
            c["mean_score"],
            c["std_score"],
            c["cvar95"],
            1.0 - c["best_freq_ws"],  # invert so lower = better for consistent display
        ])

    # Normalize within each metric for fair comparison
    metric_values = np.array(metric_values)
    metric_values_norm = np.zeros_like(metric_values)
    for j in range(4):
        vmin, vmax = metric_values[:, j].min(), metric_values[:, j].max()
        denom = vmax - vmin if vmax > vmin else 1.0
        metric_values_norm[:, j] = (metric_values[:, j] - vmin) / denom

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(4)
    width = 0.25
    colors = ["#2c7bb6", "#fdae61", "#d7191c", "#008837"]

    for i, (c, label) in enumerate(zip(candidates_to_plot, labels)):
        offset = (i - (n_candidates - 1) / 2) * width
        bars = ax.bar(x + offset, metric_values_norm[i], width,
                     color=colors[i % len(colors)], label=label,
                     edgecolor="k", linewidth=0.5)
        # Annotate with raw values
        for j, bar in enumerate(bars):
            raw_val = metric_values[i, j]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f"{raw_val:.4f}", ha="center", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylabel("Normalized Score (lower = better)", fontsize=10)
    ax.set_title("Robustness Metrics Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.25)

    fig.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "fig_24_robustness_metrics_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PlotRobust] Saved: fig_24_robustness_metrics_comparison.png")


# ══════════════════════════════════════════════════════════════
#  Master plot function
# ══════════════════════════════════════════════════════════════
def plot_all_phase4(phase4_results):
    """
    Generate all Phase 4 figures from the results dict.
    """
    print("\n[PlotRobust] Generating Phase 4 figures...")

    scenario_df = phase4_results["scenario_df"]
    sample_analysis = phase4_results["sample_analysis"]
    sample_metrics = phase4_results["sample_metrics"]
    freq_table = phase4_results["freq_table"]

    plot_scenario_best_solutions(scenario_df, sample_analysis)
    plot_weight_sampling_best_frequency(freq_table, top_n=15)
    plot_weight_simplex_regions(sample_analysis, top_n=8)
    plot_rank_distribution_64_vs_48(sample_analysis)
    plot_score_distribution_64_vs_48(sample_analysis)
    plot_robustness_metrics_comparison(sample_metrics, sample_analysis)

    print("[PlotRobust] All Phase 4 figures generated.")
