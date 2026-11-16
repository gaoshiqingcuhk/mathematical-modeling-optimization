"""
Phase 4: Weight Robustness Analysis
权重敏感性分析、Dirichlet采样、鲁棒设计方案选择
"""
import numpy as np
import pandas as pd
import os

from src.config import (
    INPUT_COLUMNS, OUTPUT_COLUMNS, OUTPUT_NAMES_CN,
    OUTPUT_TABLES, OUTPUT_FIGURES, CV_RANDOM_SEED,
)
from src.utils import ensure_dir

# ── Constants ──
N_WEIGHT_SAMPLES = 10000
DIRICHLET_ALPHA = (1.0, 1.0, 1.0)
DIRICHLET_SEED = 2026
SURROGATE_BATCH_SIZE = 250  # batch size for TOPSIS in surrogate space
WS_BATCH_SIZE = 500         # batch size for WS in surrogate space


# ══════════════════════════════════════════════════════════════
#  1. Weight Scenario Definitions
# ══════════════════════════════════════════════════════════════
def get_weight_scenarios():
    """Return list of (name, w1, w2, w3, description)"""
    return [
        ("balanced",              1/3,  1/3,  1/3,  "均衡场景"),
        ("cooling_priority",      0.60, 0.20, 0.20, "极限散热场景"),
        ("low_pumping_power",     0.20, 0.60, 0.20, "低泵功耗场景"),
        ("reliability_priority",  0.20, 0.20, 0.60, "高可靠性/低热点场景"),
        ("cooling_reliability",   0.45, 0.10, 0.45, "散热与可靠性优先"),
        ("energy_reliability",    0.10, 0.45, 0.45, "能耗与可靠性优先"),
        ("thermal_energy",        0.45, 0.45, 0.10, "散热与压降折中"),
    ]


# ══════════════════════════════════════════════════════════════
#  2. Dirichlet Weight Sampling
# ══════════════════════════════════════════════════════════════
def sample_weights_dirichlet(n=N_WEIGHT_SAMPLES, alpha=DIRICHLET_ALPHA,
                            seed=DIRICHLET_SEED):
    """Sample n weight vectors from Dirichlet(alpha) on the simplex."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(alpha, n)  # (n, 3)
    return weights


# ══════════════════════════════════════════════════════════════
#  3. Normalization
# ══════════════════════════════════════════════════════════════
def normalize_Y(Y):
    """
    Min-max normalize Y to [0, 1]. Lower is always better.
    Returns (Z, y_min, y_max, denom).
    """
    y_min = Y.min(axis=0)
    y_max = Y.max(axis=0)
    denom = y_max - y_min
    denom[denom == 0] = 1e-12
    Z = (Y - y_min) / denom
    return Z, y_min, y_max, denom


# ══════════════════════════════════════════════════════════════
#  4. Weighted Sum Scores (vectorized)
# ══════════════════════════════════════════════════════════════
def compute_ws_scores(Z, weights):
    """
    Z: (n, 3) normalized objectives
    weights: (k, 3) weight vectors
    Returns: scores (n, k) — lower is better
    """
    return Z @ weights.T  # (n, k)


# ══════════════════════════════════════════════════════════════
#  5. TOPSIS Scores (vectorized, theoretical ideal)
# ══════════════════════════════════════════════════════════════
def compute_topsis_scores_batched(Z, weights, batch_size=200):
    """
    Compute TOPSIS scores for all candidates × all weights.
    Uses theoretical ideal point (0,0,0) and anti-ideal (1,1,1)
    in Z-space, consistent with Phase 3 normalize_and_score.

    Z: (n, 3) normalized objectives [0,1], smaller is better
    weights: (k, 3) weight vectors
    batch_size: how many weights to process at once

    Returns: C (n, k) — higher is better
    """
    n, m = Z.shape
    k = weights.shape[0]
    C_all = np.zeros((n, k))

    for start in range(0, k, batch_size):
        end = min(start + batch_size, k)
        w_batch = weights[start:end]  # (b, 3)
        b = w_batch.shape[0]

        # V: (n, 3, b) — weighted normalized matrix
        V = Z[:, :, None] * w_batch.T[None, :, :]  # (n, 3, b)

        # D_plus from ideal (0,0,0)
        D_plus = np.sqrt((V ** 2).sum(axis=1))  # (n, b)

        # D_minus from anti-ideal (1,1,1) — consistent with Phase 3 convention
        # Phase 3 uses z_minus = (1,1,1) in V-space for all weights
        ones = np.ones((1, 3, b))
        D_minus = np.sqrt(((V - ones) ** 2).sum(axis=1))  # (n, b)

        denom = D_plus + D_minus
        denom[denom == 0] = 1e-12
        C_all[:, start:end] = D_minus / denom

    return C_all


# ══════════════════════════════════════════════════════════════
#  6. Core Analysis: evaluate one candidate set
# ══════════════════════════════════════════════════════════════
def analyze_candidate_set(Y, candidate_labels, candidate_X,
                          weights, key_indices=None):
    """
    Analyze a candidate set under many weight vectors.

    Parameters
    ----------
    Y : (n, 3) array — objective values
    candidate_labels : (n,) array — labels (sample IDs or candidate IDs)
    candidate_X : (n, 3) array — design variables (x1, x2, x3)
    weights : (k, 3) array — weight vectors
    key_indices : (n_key,) array or None — indices of key candidates to
                  track in detail. If None, tracks all.

    Returns
    -------
    dict with comprehensive results.
    """
    n, m = Y.shape
    k = weights.shape[0]

    # Normalize Y
    Z, y_min, y_max, denom = normalize_Y(Y)

    # ── Weighted Sum ──────────────────────────
    print(f"  Computing WS scores ({n} candidates × {k} weights)...")
    ws_scores = compute_ws_scores(Z, weights)  # (n, k), lower is better

    # Best WS candidate per weight
    best_ws_idx = np.argmin(ws_scores, axis=0)  # (k,)

    # Ranks (1-indexed, lower rank = better)
    ws_ranks = np.argsort(np.argsort(ws_scores, axis=0), axis=0) + 1  # (n, k)

    # ── TOPSIS ────────────────────────────────
    batch_size = 200 if n <= 200 else SURROGATE_BATCH_SIZE
    print(f"  Computing TOPSIS scores ({n} candidates × {k} weights, batch={batch_size})...")
    topsis_scores = compute_topsis_scores_batched(Z, weights, batch_size=batch_size)  # (n, k), higher is better

    # Best TOPSIS candidate per weight
    best_topsis_idx = np.argmax(topsis_scores, axis=0)  # (k,)

    # Ranks (1-indexed, lower rank = better since higher C = better)
    topsis_ranks = n - np.argsort(np.argsort(topsis_scores, axis=0), axis=0)  # (n, k), 1 = best

    # ── Key candidates detail ──────────────────
    if key_indices is None:
        key_indices = np.arange(n)
    key_indices = np.asarray(key_indices)

    result = {
        "n": n,
        "k": k,
        "Z": Z,
        "y_min": y_min,
        "y_max": y_max,
        "candidate_labels": candidate_labels,
        "candidate_X": candidate_X,
        "weights": weights,
        "ws_scores": ws_scores,          # (n, k)
        "topsis_scores": topsis_scores,  # (n, k)
        "ws_ranks": ws_ranks,            # (n, k)
        "topsis_ranks": topsis_ranks,    # (n, k)
        "best_ws_idx": best_ws_idx,      # (k,)
        "best_topsis_idx": best_topsis_idx,  # (k,)
        "key_indices": key_indices,
    }
    return result


# ══════════════════════════════════════════════════════════════
#  7. Robustness Metrics
# ══════════════════════════════════════════════════════════════
def compute_robustness_metrics(analysis_result, key_indices=None):
    """
    Compute robustness metrics for specified candidates.

    Parameters
    ----------
    analysis_result : dict from analyze_candidate_set
    key_indices : (n_key,) array of indices to evaluate

    Returns
    -------
    DataFrame with columns: candidate_id, x1, x2, x3, [y1, y2, y3],
    mean_score, std_score, worst_score, cvar95, best_freq_ws,
    best_freq_topsis, mean_rank_ws, std_rank_ws, mean_rank_topsis,
    std_rank_topsis, robust_score
    """
    if key_indices is None:
        key_indices = analysis_result["key_indices"]
    key_indices = np.asarray(key_indices)

    ws_scores = analysis_result["ws_scores"]        # (n, k)
    topsis_scores = analysis_result["topsis_scores"] # (n, k)
    ws_ranks = analysis_result["ws_ranks"]           # (n, k)
    topsis_ranks = analysis_result["topsis_ranks"]   # (n, k)
    best_ws_idx = analysis_result["best_ws_idx"]     # (k,)
    best_topsis_idx = analysis_result["best_topsis_idx"]  # (k,)
    labels = analysis_result["candidate_labels"]
    X = analysis_result["candidate_X"]
    k = analysis_result["k"]

    n_key = len(key_indices)
    rows = []

    for idx in key_indices:
        label = labels[idx]

        # WS scores for this candidate
        ws_i = ws_scores[idx, :]      # (k,)
        topsis_i = topsis_scores[idx, :]  # (k,)
        ws_rank_i = ws_ranks[idx, :]      # (k,)
        topsis_rank_i = topsis_ranks[idx, :]  # (k,)

        # Basic stats
        mean_score = float(np.mean(ws_i))
        std_score = float(np.std(ws_i))
        worst_score = float(np.max(ws_i))

        # CVaR 95%: average of worst 5% WS scores
        n_worst = max(1, int(k * 0.05))
        cvar95 = float(np.mean(np.sort(ws_i)[-n_worst:]))

        # Best frequency (WS)
        best_freq_ws = float(np.mean(best_ws_idx == idx))

        # Best frequency (TOPSIS)
        best_freq_topsis = float(np.mean(best_topsis_idx == idx))

        # Rank stats
        mean_rank_ws = float(np.mean(ws_rank_i))
        std_rank_ws = float(np.std(ws_rank_i))
        mean_rank_topsis = float(np.mean(topsis_rank_i))
        std_rank_topsis = float(np.std(topsis_rank_i))

        rows.append({
            "candidate": str(label),
            "x1": X[idx, 0],
            "x2": X[idx, 1],
            "x3": int(X[idx, 2]),
            "mean_score": mean_score,
            "std_score": std_score,
            "worst_score": worst_score,
            "cvar95": cvar95,
            "best_freq_ws": best_freq_ws,
            "best_freq_topsis": best_freq_topsis,
            "mean_rank_ws": mean_rank_ws,
            "std_rank_ws": std_rank_ws,
            "mean_rank_topsis": mean_rank_topsis,
            "std_rank_topsis": std_rank_topsis,
        })

    metrics_df = pd.DataFrame(rows)

    # ── Compute RobustScore ──────────────────────
    # Normalize each metric to [0,1] where 0 = best
    def norm_lower_better(x):
        """Normalize: lower x -> lower score (better)"""
        x = np.array(x)
        denom = x.max() - x.min()
        if denom < 1e-12:
            return np.zeros_like(x)
        return (x - x.min()) / denom

    # BestFreq: higher is better -> normalize such that higher freq → larger norm
    # Then -0.10 * (larger norm) = more negative → lower RobustScore → better
    best_freq_combined = (metrics_df["best_freq_ws"] + metrics_df["best_freq_topsis"]) / 2.0
    freq_norm = norm_lower_better(best_freq_combined.values)  # high freq → close to 1

    metrics_df["robust_score"] = (
        0.40 * norm_lower_better(metrics_df["mean_score"].values)
        + 0.25 * norm_lower_better(metrics_df["std_score"].values)
        + 0.25 * norm_lower_better(metrics_df["cvar95"].values)
        - 0.10 * freq_norm  # high BestFreq → large negative contribution → lower (better) score
    )

    # Rank by robust_score (lower is better)
    metrics_df["robust_rank"] = metrics_df["robust_score"].rank(method="min").astype(int)

    return metrics_df.sort_values("robust_score")


# ══════════════════════════════════════════════════════════════
#  8. Scenario Weights Analysis
# ══════════════════════════════════════════════════════════════
def analyze_scenario_weights(analysis_result, scenario_weights, scenario_names,
                             key_sample_ids=(64, 48, 13, 80)):
    """
    For each scenario weight, find best WS and TOPSIS candidates,
    and report ranks of key samples.

    Returns DataFrame for table_14.
    """
    Z = analysis_result["Z"]
    labels = analysis_result["candidate_labels"]
    X = analysis_result["candidate_X"]
    n = analysis_result["n"]

    rows = []
    for i, (name, w1, w2, w3, desc) in enumerate(scenario_weights):
        w = np.array([w1, w2, w3])

        # WS
        ws = Z @ w  # (n,)
        best_ws_i = int(np.argmin(ws))
        ws_rank_order = np.argsort(ws)

        # TOPSIS
        # V = Z * w, use Phase 3 convention: z_minus = (1,1,1)
        V = Z * w[None, :]
        D_plus = np.sqrt((V ** 2).sum(axis=1))
        D_minus = np.sqrt(((V - np.ones(3)) ** 2).sum(axis=1))
        denom = D_plus + D_minus + 1e-12
        C = D_minus / denom
        best_topsis_i = int(np.argmax(C))
        topsis_rank_order = np.argsort(C)[::-1]

        # Ranks of key samples
        row = {
            "scenario": name,
            "description": desc,
            "w1": w1, "w2": w2, "w3": w3,
            "best_ws_candidate": str(labels[best_ws_i]),
            "best_ws_x1": X[best_ws_i, 0],
            "best_ws_x2": X[best_ws_i, 1],
            "best_ws_x3": int(X[best_ws_i, 2]),
            "best_topsis_candidate": str(labels[best_topsis_i]),
            "best_topsis_x1": X[best_topsis_i, 0],
            "best_topsis_x2": X[best_topsis_i, 1],
            "best_topsis_x3": int(X[best_topsis_i, 2]),
        }

        for sid in key_sample_ids:
            mask = labels == str(sid)
            if mask.any():
                idx = np.where(mask)[0][0]
                ws_rank = int(np.where(ws_rank_order == idx)[0][0]) + 1
                topsis_rank = int(np.where(topsis_rank_order == idx)[0][0]) + 1
            else:
                ws_rank = -1
                topsis_rank = -1
            row[f"rank_{sid}_ws"] = ws_rank
            row[f"rank_{sid}_topsis"] = topsis_rank

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  9. Best Frequency Table (table_15)
# ══════════════════════════════════════════════════════════════
def build_best_frequency_table(analysis_result, metrics_df, top_n=30):
    """
    Build table of candidates sorted by best frequency.
    Includes all candidates that were ever optimal, plus additional top performers.
    """
    k = analysis_result["k"]
    labels = analysis_result["candidate_labels"]
    X = analysis_result["candidate_X"]

    # Candidates sorted by combined best frequency
    freq_order = metrics_df.sort_values("best_freq_ws", ascending=False)

    rows = []
    for _, r in freq_order.head(top_n).iterrows():
        rows.append({
            "candidate": r["candidate"],
            "x1": r["x1"], "x2": r["x2"], "x3": int(r["x3"]),
            "best_freq_ws": round(r["best_freq_ws"], 6),
            "best_freq_topsis": round(r["best_freq_topsis"], 6),
            "mean_rank_ws": round(r["mean_rank_ws"], 2),
            "std_rank_ws": round(r["std_rank_ws"], 2),
            "mean_rank_topsis": round(r["mean_rank_topsis"], 2),
            "std_rank_topsis": round(r["std_rank_topsis"], 2),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  10. Solution #64 vs #48 Comparison (table_17)
# ══════════════════════════════════════════════════════════════
def compare_64_vs_48(analysis_result, metrics_df):
    """
    Build detailed comparison table for #64 vs #48.
    """
    labels = analysis_result["candidate_labels"]
    k = analysis_result["k"]

    idx_64 = np.where(labels == "64")[0]
    idx_48 = np.where(labels == "48")[0]

    if len(idx_64) == 0 or len(idx_48) == 0:
        print("[WARN] #64 or #48 not found in candidate set!")
        return pd.DataFrame()

    idx_64 = idx_64[0]
    idx_48 = idx_48[0]

    ws_64 = analysis_result["ws_scores"][idx_64, :]
    ws_48 = analysis_result["ws_scores"][idx_48, :]
    rank_ws_64 = analysis_result["ws_ranks"][idx_64, :]
    rank_ws_48 = analysis_result["ws_ranks"][idx_48, :]
    rank_topsis_64 = analysis_result["topsis_ranks"][idx_64, :]
    rank_topsis_48 = analysis_result["topsis_ranks"][idx_48, :]
    best_ws_idx = analysis_result["best_ws_idx"]
    best_topsis_idx = analysis_result["best_topsis_idx"]

    def compute_cvar95(scores):
        n_worst = max(1, int(k * 0.05))
        return float(np.mean(np.sort(scores)[-n_worst:]))

    def explain_better(val_64, val_48, lower_is_better=True):
        if lower_is_better:
            if val_64 < val_48:
                return "#64", "#64 has lower (better) value"
            elif val_48 < val_64:
                return "#48", "#48 has lower (better) value"
            else:
                return "Tie", "Equal values"
        else:
            if val_64 > val_48:
                return "#64", "#64 has higher (better) value"
            elif val_48 > val_64:
                return "#48", "#48 has higher (better) value"
            else:
                return "Tie", "Equal values"

    metrics = [
        ("Weight Sampling WS Best Freq",
         float(np.mean(best_ws_idx == idx_64)),
         float(np.mean(best_ws_idx == idx_48)),
         True, "higher is better"),
        ("Weight Sampling TOPSIS Best Freq",
         float(np.mean(best_topsis_idx == idx_64)),
         float(np.mean(best_topsis_idx == idx_48)),
         True, "higher is better"),
        ("Mean WS Score",
         float(np.mean(ws_64)), float(np.mean(ws_48)),
         False, "lower is better"),
        ("WS Score Std",
         float(np.std(ws_64)), float(np.std(ws_48)),
         False, "lower is better (more stable)"),
        ("CVaR95 (Worst 5% WS)",
         compute_cvar95(ws_64), compute_cvar95(ws_48),
         False, "lower is better"),
        ("Mean WS Rank",
         float(np.mean(rank_ws_64)), float(np.mean(rank_ws_48)),
         False, "lower is better"),
        ("WS Rank Std",
         float(np.std(rank_ws_64)), float(np.std(rank_ws_48)),
         False, "lower is better"),
        ("Worst WS Rank",
         float(np.max(rank_ws_64)), float(np.max(rank_ws_48)),
         False, "lower is better"),
        ("Mean TOPSIS Rank",
         float(np.mean(rank_topsis_64)), float(np.mean(rank_topsis_48)),
         False, "lower is better"),
        ("TOPSIS Rank Std",
         float(np.std(rank_topsis_64)), float(np.std(rank_topsis_48)),
         False, "lower is better"),
    ]

    rows = []
    for name, v64, v48, higher_better, default_explanation in metrics:
        better, explanation = explain_better(v64, v48, lower_is_better=not higher_better)
        rows.append({
            "metric": name,
            "sample_64": round(v64, 6),
            "sample_48": round(v48, 6),
            "better_solution": better,
            "explanation": explanation if better != "Tie" else default_explanation,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  11. Robust Design Recommendation (table_18)
# ══════════════════════════════════════════════════════════════
def build_robust_recommendation(metrics_df, analysis_result):
    """
    Build final recommendation table based on RobustScore ranking.
    """
    df = metrics_df.sort_values("robust_score").head(10).copy()

    rows = []
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        sol_type = "comprehensive_optimum" if r["candidate"] == "64" else \
                   ("robustness_candidate" if r["candidate"] == "48" else \
                    ("min_thermal" if r["candidate"] == "13" else \
                     ("min_pressure" if r["candidate"] == "80" else "other")))

        # Interpret recommendation
        if rank == 1:
            if r["candidate"] == "64":
                reason = "综合最优方案，等权和鲁棒双重验证下均为最优"
            elif r["candidate"] == "48":
                reason = "鲁棒最优方案，对权重变化容忍度最高"
            else:
                reason = f"鲁棒评分最优方案，综合Mean/Std/CVaR/BestFreq最优"
        elif r["candidate"] == "64":
            reason = "综合最优方案，平均性能最佳"
        elif r["candidate"] == "48":
            reason = "鲁棒备选方案，稳定性优异"
        else:
            reason = ""

        rows.append({
            "solution_type": sol_type,
            "candidate_id": r["candidate"],
            "x1": r["x1"], "x2": r["x2"], "x3": int(r["x3"]),
            "mean_score": round(r["mean_score"], 6),
            "std_score": round(r["std_score"], 6),
            "cvar95": round(r["cvar95"], 6),
            "best_freq_ws": round(r["best_freq_ws"], 6),
            "robust_score": round(r["robust_score"], 6),
            "robust_rank": int(r["robust_rank"]),
            "recommendation_reason": reason,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  12. Main Phase 4 Runner
# ══════════════════════════════════════════════════════════════
def run_phase4(df, p3_results):
    """
    Main Phase 4 execution.

    Parameters
    ----------
    df : pd.DataFrame — clean data (84 samples)
    p3_results : dict — from phase3(), includes surrogate_results

    Returns
    -------
    dict of all Phase 4 results
    """
    print("\n" + "=" * 60)
    print("Phase 4: Weight Robustness Analysis")
    print("=" * 60)

    ensure_dir(OUTPUT_TABLES)

    # ── Prepare scenario weights ──
    scenario_list = get_weight_scenarios()
    scenario_weights = [(name, w1, w2, w3, desc) for name, w1, w2, w3, desc in scenario_list]

    # ── Sample Dirichlet weights ──
    print(f"\n[Phase4] Sampling {N_WEIGHT_SAMPLES} weights from Dirichlet{tuple(DIRICHLET_ALPHA)}...")
    dirichlet_weights = sample_weights_dirichlet(N_WEIGHT_SAMPLES, DIRICHLET_ALPHA, DIRICHLET_SEED)
    print(f"  Weight range: w1=[{dirichlet_weights[:,0].min():.3f}, {dirichlet_weights[:,0].max():.3f}], "
          f"w2=[{dirichlet_weights[:,1].min():.3f}, {dirichlet_weights[:,1].max():.3f}], "
          f"w3=[{dirichlet_weights[:,2].min():.3f}, {dirichlet_weights[:,2].max():.3f}]")
    print(f"  Sum check: all sums within [1-1e-10, 1+1e-10]: "
          f"{np.allclose(dirichlet_weights.sum(axis=1), 1.0)}")

    # ══════════════════════════════════════════════════════════
    #  SAMPLE SPACE ANALYSIS (80 enhanced designs)
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("[Phase4] Sample Space Robustness Analysis (80 enhanced designs)")
    print("-" * 50)

    df_enhanced = df[(df["pin_width_ratio"] > 0) & (df["pin_rows"] >= 2)].copy()
    Y_sample = df_enhanced[OUTPUT_COLUMNS].values
    X_sample = df_enhanced[INPUT_COLUMNS].values
    labels_sample = df_enhanced["sample_id"].values.astype(str)

    # Identify key sample indices
    key_ids = [64, 48, 13, 80]
    key_indices_sample = []
    for sid in key_ids:
        mask = labels_sample == str(sid)
        if mask.any():
            key_indices_sample.append(int(np.where(mask)[0][0]))

    # Add Pareto samples
    from src.optimization import pareto_frontier
    pareto_mask_sample = pareto_frontier(Y_sample)
    pareto_indices_sample = list(np.where(pareto_mask_sample)[0])
    for pi in pareto_indices_sample:
        if pi not in key_indices_sample:
            key_indices_sample.append(pi)

    key_indices_sample = np.array(sorted(set(key_indices_sample)))
    print(f"  Tracking {len(key_indices_sample)} key candidates (incl. Pareto, #64, #48, #13, #80)")

    # Analyze sample space
    sample_analysis = analyze_candidate_set(
        Y_sample, labels_sample, X_sample,
        dirichlet_weights, key_indices=key_indices_sample
    )

    # ── Scenario analysis (sample space) ──
    print("\n[Phase4] Scenario weight analysis (sample space)...")
    scenario_df = analyze_scenario_weights(
        sample_analysis, scenario_weights, [s[0] for s in scenario_list]
    )
    path = os.path.join(OUTPUT_TABLES, "table_14_scenario_weight_results.csv")
    scenario_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_14_scenario_weight_results.csv")

    # ── Robustness metrics (sample space) ──
    print("\n[Phase4] Computing robustness metrics (sample space)...")
    # Track all candidates that were ever optimal, plus key candidates
    best_ws_idx = sample_analysis["best_ws_idx"]
    best_topsis_idx = sample_analysis["best_topsis_idx"]
    ever_optimal = np.unique(np.concatenate([best_ws_idx, best_topsis_idx]))
    all_tracked = np.unique(np.concatenate([key_indices_sample, ever_optimal]))
    print(f"  Total tracked candidates: {len(all_tracked)} (key={len(key_indices_sample)}, ever_optimal={len(ever_optimal)})")

    sample_metrics = compute_robustness_metrics(sample_analysis, all_tracked)
    path = os.path.join(OUTPUT_TABLES, "table_16_robustness_metrics.csv")
    sample_metrics.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_16_robustness_metrics.csv ({len(sample_metrics)} rows)")

    # Print top 5 by RobustScore
    print("\n  Top 5 by RobustScore (sample space):")
    for _, r in sample_metrics.head(5).iterrows():
        print(f"    #{r['candidate']}: x=({r['x1']:.2f},{r['x2']:.2f},{int(r['x3'])}), "
              f"RobustScore={r['robust_score']:.6f}, rank={int(r['robust_rank'])}")

    # ── Best frequency table ──
    freq_table = build_best_frequency_table(sample_analysis, sample_metrics, top_n=30)
    path = os.path.join(OUTPUT_TABLES, "table_15_weight_sampling_best_frequency.csv")
    freq_table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_15_weight_sampling_best_frequency.csv")

    # ── #64 vs #48 comparison ──
    print("\n[Phase4] Building #64 vs #48 comparison...")
    comparison_64_48 = compare_64_vs_48(sample_analysis, sample_metrics)
    path = os.path.join(OUTPUT_TABLES, "table_17_solution_64_vs_48_robustness.csv")
    comparison_64_48.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_17_solution_64_vs_48_robustness.csv")

    # Print key comparison
    if len(comparison_64_48) > 0:
        print("\n  #64 vs #48 Key Comparison:")
        for _, r in comparison_64_48.iterrows():
            print(f"    {r['metric']}: #64={r['sample_64']}, #48={r['sample_48']} → {r['better_solution']}")

    # ── Robust recommendation ──
    print("\n[Phase4] Building robust design recommendation...")
    recommendation = build_robust_recommendation(sample_metrics, sample_analysis)
    path = os.path.join(OUTPUT_TABLES, "table_18_robust_design_recommendation.csv")
    recommendation.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_18_robust_design_recommendation.csv")

    # ══════════════════════════════════════════════════════════
    #  SURROGATE SPACE — Save full candidates CSV
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("[Phase4] Surrogate Space: Saving full candidate data")
    print("-" * 50)

    surrogate_results = p3_results["surrogate_results"]
    X_cand = surrogate_results["X_cand"]
    Y_pred = surrogate_results["Y_pred"]

    # Save table_10b: all surrogate candidates
    all_cand_rows = []
    for i in range(len(X_cand)):
        all_cand_rows.append({
            "candidate_id": i,
            "x1": round(float(X_cand[i, 0]), 6),
            "x2": round(float(X_cand[i, 1]), 6),
            "x3": int(X_cand[i, 2]),
            "pred_y1": float(Y_pred[i, 0]),
            "pred_y2": float(Y_pred[i, 1]),
            "pred_y3": float(Y_pred[i, 2]),
        })

    all_cand_df = pd.DataFrame(all_cand_rows)
    path = os.path.join(OUTPUT_TABLES, "table_10b_surrogate_search_all_candidates.csv")
    all_cand_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_10b_surrogate_search_all_candidates.csv ({len(all_cand_df)} rows)")

    # ══════════════════════════════════════════════════════════
    #  SURROGATE SPACE — Robustness on Pareto subset
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("[Phase4] Surrogate Space Robustness (Pareto subset only)")
    print("-" * 50)

    # Only analyze Pareto candidates (sufficient for optimality)
    pareto_mask_surr = surrogate_results["pareto_mask"]
    pareto_indices = np.where(pareto_mask_surr)[0]
    n_pareto = len(pareto_indices)
    print(f"  Surrogate Pareto candidates: {n_pareto}")

    Y_pareto = Y_pred[pareto_indices]
    X_pareto = X_cand[pareto_indices]
    labels_pareto = np.array([f"S{i}" for i in pareto_indices])  # S-prefix for surrogate

    # For surrogate space, track all Pareto (manageable at ~6389)
    surr_analysis = analyze_candidate_set(
        Y_pareto, labels_pareto, X_pareto,
        dirichlet_weights, key_indices=np.arange(n_pareto)
    )

    # Find nearest surrogate candidate to key samples
    def find_nearest_surrogate(sample_id):
        row = df[df["sample_id"] == sample_id]
        if len(row) == 0:
            return None
        x_target = row[INPUT_COLUMNS].values[0]
        diff = X_pareto - x_target[None, :]
        diff[:, 0] /= 0.30
        diff[:, 1] /= 1.5
        diff[:, 2] /= 10.0
        dist = np.sqrt((diff ** 2).sum(axis=1))
        idx = int(np.argmin(dist))
        return pareto_indices[idx], float(dist.min())

    nearest_info = {}
    for sid in [64, 48, 13, 80]:
        result_near = find_nearest_surrogate(sid)
        if result_near is not None:
            nearest_info[sid] = result_near
            print(f"  Nearest surrogate to #{sid}: candidate S{result_near[0]}, dist={result_near[1]:.4f}")

    # ── Best frequency in surrogate space ──
    best_ws_idx_surr = surr_analysis["best_ws_idx"]       # indices into Pareto subset
    best_topsis_idx_surr = surr_analysis["best_topsis_idx"]

    # Count frequencies
    from collections import Counter
    ws_freq = Counter(best_ws_idx_surr)
    topsis_freq = Counter(best_topsis_idx_surr)

    # Top surrogate candidates by frequency
    print(f"\n  Top 5 surrogate candidates by WS best frequency:")
    for cand_idx, count in ws_freq.most_common(5):
        pct = 100.0 * count / N_WEIGHT_SAMPLES
        x = X_pareto[cand_idx]
        print(f"    S{pareto_indices[cand_idx]}: x=({x[0]:.4f},{x[1]:.4f},{int(x[2])}), "
              f"freq={pct:.2f}%")

    print(f"\n  Top 5 surrogate candidates by TOPSIS best frequency:")
    for cand_idx, count in topsis_freq.most_common(5):
        pct = 100.0 * count / N_WEIGHT_SAMPLES
        x = X_pareto[cand_idx]
        print(f"    S{pareto_indices[cand_idx]}: x=({x[0]:.4f},{x[1]:.4f},{int(x[2])}), "
              f"freq={pct:.2f}%")

    # ── Robustness metrics for surrogate Pareto ──
    surr_metrics = compute_robustness_metrics(surr_analysis, np.arange(n_pareto))
    # Map back to original candidate IDs
    surr_metrics["candidate"] = [f"S{pareto_indices[i]}" for i in range(n_pareto)]
    surr_metrics["original_candidate_id"] = [int(pareto_indices[i]) for i in range(n_pareto)]

    # ══════════════════════════════════════════════════════════
    #  PACKAGE RESULTS
    # ══════════════════════════════════════════════════════════
    results = {
        "scenario_df": scenario_df,
        "sample_analysis": sample_analysis,
        "sample_metrics": sample_metrics,
        "freq_table": freq_table,
        "comparison_64_48": comparison_64_48,
        "recommendation": recommendation,
        "surrogate_analysis": surr_analysis,
        "surrogate_metrics": surr_metrics,
        "dirichlet_weights": dirichlet_weights,
        "nearest_info": nearest_info,
        "all_candidates_df": all_cand_df,
    }

    print("\n[Phase4] Done!")
    return results
