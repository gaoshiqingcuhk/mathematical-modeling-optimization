"""
多目标优化模块
Pareto 非支配排序、网格搜索、加权和、TOPSIS
"""
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV

from src.config import (
    INPUT_COLUMNS, OUTPUT_COLUMNS, OUTPUT_NAMES_CN,
    OUTPUT_TABLES, OUTPUT_MODELS, CV_RANDOM_SEED,
)
from src.utils import ensure_dir


# ────────────────────────────────────────────────────────────
#  Pareto 非支配排序
# ────────────────────────────────────────────────────────────
def pareto_frontier(Y: np.ndarray) -> np.ndarray:
    """
    找到 Pareto 前沿（非支配解）的索引。
    Y: (n_samples, n_objectives)，越小越好。
    返回布尔掩码，True 表示 Pareto 最优。

    算法：先按第一目标排序，然后与已确认的 Pareto 集对比，
    复杂度 O(n log n + n * P)，其中 P 为 Pareto 集大小（通常 << n）。
    """
    n = Y.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    # 按第一目标升序排列
    order = np.argsort(Y[:, 0])
    is_pareto = np.ones(n, dtype=bool)
    pareto_list = []  # 已确认的 Pareto 解索引

    for pos in range(n):
        idx = order[pos]
        yi = Y[idx]
        dominated = False
        for pj in pareto_list:
            yj = Y[pj]
            # pj 在 obj1 上已 ≤ yi，只需检查 obj2+ 是否全 ≤ 且至少一个 <
            if np.all(yj[1:] <= yi[1:]) and np.any(yj[1:] < yi[1:]):
                dominated = True
                break
        if not dominated:
            pareto_list.append(idx)
        else:
            is_pareto[idx] = False

    return is_pareto


# ────────────────────────────────────────────────────────────
#  归一化与评分
# ────────────────────────────────────────────────────────────
def normalize_and_score(Y: np.ndarray) -> tuple:
    """
    对 Y 做 min-max 归一化，返回 (Z, ws_scores, topsis_scores)
    ws_scores 越小越好，topsis_scores 越大越好。
    """
    y_min = Y.min(axis=0)
    y_max = Y.max(axis=0)
    denom = y_max - y_min
    denom[denom == 0] = 1e-12
    Z = (Y - y_min) / denom

    # 等权加权和
    w = np.array([1/3, 1/3, 1/3])
    ws_scores = Z @ w  # 越小越好

    # TOPSIS
    V = Z * w  # 加权归一化
    z_plus = np.zeros(3)   # min of each col is 0 (ideally)
    z_minus = np.ones(3)   # max of each col is 1

    D_plus = np.sqrt(np.sum((V - z_plus) ** 2, axis=1))
    D_minus = np.sqrt(np.sum((V - z_minus) ** 2, axis=1))
    denom_t = D_plus + D_minus
    denom_t[denom_t == 0] = 1e-12
    topsis_scores = D_minus / denom_t  # 越大越好

    return Z, ws_scores, topsis_scores


# ────────────────────────────────────────────────────────────
#  样本空间优化
# ────────────────────────────────────────────────────────────
def sample_space_optimization(df: pd.DataFrame) -> dict:
    """
    基于 84 个真实样本的多目标优化。
    排除无针肋基准结构 (pin_width_ratio=0, pin_rows=0) 仅做对照。
    """
    print("\n" + "-" * 50)
    print("[Opt] Sample Space Optimization")
    print("-" * 50)

    # 增强针肋结构：x1>0 且 x3>=2
    df_enhanced = df[(df["pin_width_ratio"] > 0) & (df["pin_rows"] >= 2)].copy()
    print(f"  Enhanced samples: {len(df_enhanced)} (excluded {len(df)-len(df_enhanced)} baseline)")

    Y = df_enhanced[OUTPUT_COLUMNS].values
    pareto_mask = pareto_frontier(Y)
    n_pareto = pareto_mask.sum()
    print(f"  Pareto frontier: {n_pareto} samples")

    Z, ws_scores, topsis_scores = normalize_and_score(Y)

    # 排序
    ws_rank = np.argsort(ws_scores)      # 越小越好 -> 升序
    topsis_rank = np.argsort(topsis_scores)[::-1]  # 越大越好 -> 降序

    # 构建结果 DataFrame
    result = df_enhanced.copy()
    result["weighted_sum"] = ws_scores
    result["topsis_score"] = topsis_scores
    result["ws_rank"] = np.argsort(ws_rank) + 1
    result["topsis_rank"] = np.argsort(np.argsort(topsis_scores)[::-1]) + 1
    result["is_pareto"] = pareto_mask

    # ── 保存 Pareto 表 ──
    pareto_df = result[pareto_mask].sort_values("weighted_sum")
    cols_out = ["sample_id"] + INPUT_COLUMNS + OUTPUT_COLUMNS + ["weighted_sum", "topsis_score"]
    path_pareto = os.path.join(OUTPUT_TABLES, "table_08_sample_space_pareto.csv")
    pareto_df[cols_out].to_csv(path_pareto, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_08_sample_space_pareto.csv ({len(pareto_df)} rows)")

    # ── 保存 Top 10 ──
    top10_ws = result.iloc[ws_rank[:10]]
    top10_topsis = result.iloc[topsis_rank[:10]]
    top10_combined = top10_ws[["sample_id"] + INPUT_COLUMNS + OUTPUT_COLUMNS + ["weighted_sum", "topsis_score"]].copy()
    top10_combined.insert(0, "rank_weighted_sum", range(1, 11))
    # merge topsis rank
    t_rank_map = dict(zip(top10_topsis["sample_id"], range(1, 11)))
    top10_combined["rank_topsis"] = top10_combined["sample_id"].map(t_rank_map)
    for sid in top10_combined["sample_id"]:
        if pd.isna(top10_combined.loc[top10_combined["sample_id"]==sid, "rank_topsis"].values[0]):
            # find actual topsis rank
            actual_rank = result[result["sample_id"]==sid]["topsis_rank"].values[0]
            top10_combined.loc[top10_combined["sample_id"]==sid, "rank_topsis"] = actual_rank

    path_top10 = os.path.join(OUTPUT_TABLES, "table_09_sample_space_top10.csv")
    top10_combined.to_csv(path_top10, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_09_sample_space_top10.csv")

    print(f"\n  WS best: Sample {int(result.iloc[ws_rank[0]]['sample_id'])}, "
          f"WS={ws_scores[ws_rank[0]]:.6f}")
    print(f"  TOPSIS best: Sample {int(result.iloc[topsis_rank[0]]['sample_id'])}, "
          f"C={topsis_scores[topsis_rank[0]]:.6f}")

    # 关键样本排名
    for sid in [64, 48, 13, 80]:
        if sid in result["sample_id"].values:
            r = result[result["sample_id"] == sid].iloc[0]
            print(f"  Sample {sid}: ws_rank={int(r['ws_rank'])}, topsis_rank={int(r['topsis_rank'])}, "
                  f"pareto={r['is_pareto']}")

    return {
        "result": result,
        "pareto_mask": pareto_mask,
        "ws_scores": ws_scores,
        "topsis_scores": topsis_scores,
        "ws_rank": ws_rank,
        "topsis_rank": topsis_rank,
    }


# ────────────────────────────────────────────────────────────
#  代理模型网格搜索
# ────────────────────────────────────────────────────────────
def surrogate_grid_search(df: pd.DataFrame, models: dict) -> dict:
    """
    在强化设计域内进行密集网格搜索。
    models: {target: (model_name, fitted_pipeline)}
    """
    print("\n" + "-" * 50)
    print("[Opt] Surrogate Grid Search")
    print("-" * 50)

    # 网格定义
    x1_grid = np.linspace(0.10, 0.30, 201)
    x2_grid = np.linspace(3.0, 4.5, 151)
    x3_grid = np.array([2, 4, 6, 8, 10])

    n_candidates = len(x1_grid) * len(x2_grid) * len(x3_grid)
    print(f"  Grid: {len(x1_grid)} x {len(x2_grid)} x {len(x3_grid)} = {n_candidates} candidates")

    # 构建候选矩阵
    X_cand = []
    for x3 in x3_grid:
        for x2 in x2_grid:
            for x1 in x1_grid:
                X_cand.append([x1, x2, x3])
    X_cand = np.array(X_cand)

    # 预测
    preds = {}
    for target in OUTPUT_COLUMNS:
        _, pipe = models[target]
        preds[target] = pipe.predict(X_cand)
        print(f"  Predicted {target}: min={preds[target].min():.6f}, max={preds[target].max():.6f}")

    Y_pred = np.column_stack([preds[t] for t in OUTPUT_COLUMNS])

    # Pareto
    pareto_mask = pareto_frontier(Y_pred)
    print(f"  Pareto frontier: {pareto_mask.sum()} candidates")

    # 归一化与评分
    Z, ws_scores, topsis_scores = normalize_and_score(Y_pred)

    ws_rank = np.argsort(ws_scores)
    topsis_rank = np.argsort(topsis_scores)[::-1]

    # ── 找到最近真实样本 ──
    X_real = df[(df["pin_width_ratio"] > 0) & (df["pin_rows"] >= 2)][INPUT_COLUMNS].values
    real_ids = df[(df["pin_width_ratio"] > 0) & (df["pin_rows"] >= 2)]["sample_id"].values

    def find_nearest(x_cand):
        # 加权距离: x1, x2 连续，x3 离散
        diff = np.abs(X_real - x_cand)
        diff[:, 0] /= 0.30   # normalize by range
        diff[:, 1] /= 1.5
        diff[:, 2] /= 10.0
        dist = np.sqrt(np.sum(diff**2, axis=1))
        idx = np.argmin(dist)
        return real_ids[idx], dist[idx]

    # ── 构建 Pareto 结果表 ──
    pareto_indices = np.where(pareto_mask)[0]
    pareto_rows = []
    for idx in pareto_indices:
        nearest_id, nearest_dist = find_nearest(X_cand[idx])
        pareto_rows.append({
            "candidate_id": idx,
            "x1": X_cand[idx, 0],
            "x2": X_cand[idx, 1],
            "x3": int(X_cand[idx, 2]),
            "pred_y1": Y_pred[idx, 0],
            "pred_y2": Y_pred[idx, 1],
            "pred_y3": Y_pred[idx, 2],
            "weighted_sum": ws_scores[idx],
            "topsis_score": topsis_scores[idx],
            "nearest_sample_id": int(nearest_id),
            "nearest_distance": round(float(nearest_dist), 6),
        })

    pareto_df = pd.DataFrame(pareto_rows).sort_values("weighted_sum")
    path = os.path.join(OUTPUT_TABLES, "table_10_surrogate_search_pareto.csv")
    pareto_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_10_surrogate_search_pareto.csv ({len(pareto_df)} rows)")

    # ── 构建 Top 20 结果表 ──
    ws_top20_idx = ws_rank[:20]
    top20_rows = []
    for rank, idx in enumerate(ws_top20_idx, 1):
        nearest_id, nearest_dist = find_nearest(X_cand[idx])
        # find topsis rank
        t_rank = np.where(topsis_rank == idx)[0][0] + 1
        top20_rows.append({
            "rank_weighted_sum": rank,
            "rank_topsis": int(t_rank),
            "x1": X_cand[idx, 0],
            "x2": X_cand[idx, 1],
            "x3": int(X_cand[idx, 2]),
            "pred_y1": Y_pred[idx, 0],
            "pred_y2": Y_pred[idx, 1],
            "pred_y3": Y_pred[idx, 2],
            "weighted_sum": ws_scores[idx],
            "topsis_score": topsis_scores[idx],
            "nearest_sample_id": int(nearest_id),
            "nearest_distance": round(float(nearest_dist), 6),
        })

    top20_df = pd.DataFrame(top20_rows)
    path = os.path.join(OUTPUT_TABLES, "table_11_surrogate_search_top20.csv")
    top20_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_11_surrogate_search_top20.csv")

    best_ws_idx = ws_rank[0]
    best_topsis_idx = topsis_rank[0]
    print(f"\n  WS optimum: x=({X_cand[best_ws_idx,0]:.4f}, {X_cand[best_ws_idx,1]:.4f}, {int(X_cand[best_ws_idx,2])}), "
          f"WS={ws_scores[best_ws_idx]:.6f}")
    print(f"  TOPSIS optimum: x=({X_cand[best_topsis_idx,0]:.4f}, {X_cand[best_topsis_idx,1]:.4f}, "
          f"{int(X_cand[best_topsis_idx,2])}), C={topsis_scores[best_topsis_idx]:.6f}")

    nearest_ws_id, nearest_ws_dist = find_nearest(X_cand[best_ws_idx])
    nearest_topsis_id, nearest_topsis_dist = find_nearest(X_cand[best_topsis_idx])
    print(f"  Nearest to WS optimum: Sample {int(nearest_ws_id)}, dist={nearest_ws_dist:.6f}")
    print(f"  Nearest to TOPSIS optimum: Sample {int(nearest_topsis_id)}, dist={nearest_topsis_dist:.6f}")

    return {
        "X_cand": X_cand,
        "Y_pred": Y_pred,
        "pareto_mask": pareto_mask,
        "ws_scores": ws_scores,
        "topsis_scores": topsis_scores,
        "ws_rank": ws_rank,
        "topsis_rank": topsis_rank,
        "best_ws_idx": best_ws_idx,
        "best_topsis_idx": best_topsis_idx,
        "nearest_ws_id": nearest_ws_id,
        "nearest_topsis_id": nearest_topsis_id,
    }


# ────────────────────────────────────────────────────────────
#  候选方案对比表
# ────────────────────────────────────────────────────────────
def build_candidate_comparison(
    df: pd.DataFrame,
    sample_results: dict,
    surrogate_results: dict,
    models: dict,
) -> pd.DataFrame:
    """
    构建 table_12: 综合最优候选方案对比。
    """
    print("\n" + "-" * 50)
    print("[Opt] Building candidate comparison table")
    print("-" * 50)

    X_real = df[INPUT_COLUMNS].values
    Y_real = df[OUTPUT_COLUMNS].values
    sample_ids = df["sample_id"].values

    # 使用全增强样本集做归一化参照
    df_enhanced = df[(df["pin_width_ratio"] > 0) & (df["pin_rows"] >= 2)]
    Y_ref = df_enhanced[OUTPUT_COLUMNS].values
    y_ref_min = Y_ref.min(axis=0)
    y_ref_max = Y_ref.max(axis=0)
    denom_ref = y_ref_max - y_ref_min
    denom_ref[denom_ref == 0] = 1e-12

    rows = []

    def add_row(source, sid, note=""):
        mask = sample_ids == sid
        if not mask.any():
            return
        idx = np.where(mask)[0][0]
        x = X_real[idx]
        y_true = Y_real[idx]

        # 代理模型预测
        preds = {}
        for target in OUTPUT_COLUMNS:
            _, pipe = models[target]
            preds[target] = float(pipe.predict(x.reshape(1, -1))[0])

        # 归一化（基于全增强样本集）
        z = (y_true - y_ref_min) / denom_ref
        w = np.array([1/3, 1/3, 1/3])
        ws_val = float(z @ w)
        V = z * w
        D_plus = np.sqrt(np.sum((V - np.zeros(3))**2))
        D_minus = np.sqrt(np.sum((V - np.ones(3))**2))
        topsis_val = float(D_minus / (D_plus + D_minus + 1e-12))

        rows.append({
            "source": source,
            "sample_id": int(sid),
            "x1": x[0],
            "x2": x[1],
            "x3": int(x[2]),
            "true_y1": y_true[0],
            "true_y2": y_true[1],
            "true_y3": y_true[2],
            "pred_y1": preds["thermal_resistance"],
            "pred_y2": preds["pressure_drop"],
            "pred_y3": preds["temperature_nonuniformity"],
            "weighted_sum": round(ws_val, 6),
            "topsis_score": round(topsis_val, 6),
            "note": note,
        })

    # 关键样本
    add_row("sample_64", 64, "Phase 1 equal-weight optimum")
    add_row("sample_48", 48, "Robustness candidate")
    add_row("sample_13", 13, "Min thermal resistance")
    add_row("sample_80", 80, "Min pressure drop")

    # 样本空间最优
    result_df = sample_results["result"]
    ws_best_sample = int(result_df.iloc[sample_results["ws_rank"][0]]["sample_id"])
    topsis_best_sample = int(result_df.iloc[sample_results["topsis_rank"][0]]["sample_id"])
    add_row("sample_ws_best", ws_best_sample, "Sample space WS optimum")
    add_row("sample_topsis_best", topsis_best_sample, "Sample space TOPSIS optimum")

    # 代理搜索最优的最近真实样本
    add_row("surrogate_ws_nearest", int(surrogate_results["nearest_ws_id"]),
            "Nearest to surrogate WS optimum")
    add_row("surrogate_topsis_nearest", int(surrogate_results["nearest_topsis_id"]),
            "Nearest to surrogate TOPSIS optimum")

    comp_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_TABLES, "table_12_final_candidate_comparison.csv")
    comp_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_12_final_candidate_comparison.csv ({len(comp_df)} rows)")
    return comp_df


# ────────────────────────────────────────────────────────────
#  统一 GPR 复核
# ────────────────────────────────────────────────────────────
def unified_gpr_check(df: pd.DataFrame, surrogate_results: dict) -> pd.DataFrame:
    """
    使用统一 GPR 模型复核主代理模型结果。
    """
    print("\n" + "-" * 50)
    print("[Opt] Unified GPR consistency check")
    print("-" * 50)

    X = df[INPUT_COLUMNS].values.astype(float)
    from src.surrogate_models import build_all_models

    # 训练统一 GPR
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
    gpr_pipes = {}
    for target in OUTPUT_COLUMNS:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("gpr", GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                random_state=CV_RANDOM_SEED,
            )),
        ])
        pipe.fit(X, df[target].values)
        gpr_pipes[target] = pipe
        print(f"  GPR fitted for {target}")

    # 加载主模型
    main_models = {}
    path_map = {
        "thermal_resistance": "best_model_y1.joblib",
        "pressure_drop": "best_model_y2.joblib",
        "temperature_nonuniformity": "best_model_y3.joblib",
    }
    for target in OUTPUT_COLUMNS:
        main_models[target] = joblib.load(os.path.join(OUTPUT_MODELS, path_map[target]))

    # 比较候选
    X_cand = surrogate_results["X_cand"]
    best_ws_idx = surrogate_results["best_ws_idx"]
    best_topsis_idx = surrogate_results["best_topsis_idx"]

    check_points = {
        "Sample 64": df[df["sample_id"] == 64][INPUT_COLUMNS].values[0],
        "Sample 48": df[df["sample_id"] == 48][INPUT_COLUMNS].values[0],
        "Surrogate WS Optimum": X_cand[best_ws_idx],
        "Surrogate TOPSIS Optimum": X_cand[best_topsis_idx],
    }

    rows = []
    for label, x in check_points.items():
        x = x.reshape(1, -1)
        row = {"candidate": label, "x1": x[0, 0], "x2": x[0, 1], "x3": int(x[0, 2])}
        diffs = []
        for target in OUTPUT_COLUMNS:
            main_pred = float(main_models[target].predict(x)[0])
            gpr_pred = float(gpr_pipes[target].predict(x)[0])
            diff = abs(main_pred - gpr_pred)
            row[f"main_pred_{target.split('_')[0]}"] = main_pred
            row[f"gpr_pred_{target.split('_')[0]}"] = gpr_pred
            row[f"abs_diff_{target.split('_')[0]}"] = diff
            diffs.append(diff)

        max_diff = max(diffs)
        row["max_abs_diff"] = max_diff
        row["conclusion"] = "Consistent" if max_diff < 0.005 else (
            "Acceptable" if max_diff < 0.01 else "Discrepancy"
        )
        rows.append(row)

    check_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_TABLES, "table_13_surrogate_vs_unified_gpr_check.csv")
    check_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved: table_13_surrogate_vs_unified_gpr_check.csv")
    for _, r in check_df.iterrows():
        print(f"  {r['candidate']}: max_abs_diff={r['max_abs_diff']:.6f} -> {r['conclusion']}")

    return check_df
