"""
模型选择与排序
"""
import numpy as np
import pandas as pd
import json
import os
from src.config import (
    OUTPUT_COLUMNS, OUTPUT_NAMES_CN, OUTPUT_TABLES, OUTPUT_MODELS,
    MODEL_SELECTION_JSON,
)
from src.utils import ensure_dir


def rank_models(cv_df: pd.DataFrame) -> dict:
    """
    根据综合评分对模型排序。

    对每个目标 y_j，将各模型的 RMSE_mean 做 min-max 归一化：
      norm_RMSE_j = (RMSE_j - min_RMSE_j) / (max_RMSE_j - min_RMSE_j)
    综合评分 = mean(norm_RMSE_y1, norm_RMSE_y2, norm_RMSE_y3)
    分数越低越好。
    """
    models_list = sorted(cv_df["model"].unique())

    # 构建 (model, target) -> RMSE_mean 映射
    rmse_map = {}
    for _, row in cv_df.iterrows():
        rmse_map[(row["model"], row["target"])] = row["RMSE_mean"]

    # min-max normalize per target
    norm_rmse = {}
    for target in OUTPUT_COLUMNS:
        values = np.array([rmse_map[(m, target)] for m in models_list])
        vmin, vmax = values.min(), values.max()
        denom = vmax - vmin if vmax > vmin else 1.0
        for m in models_list:
            norm_rmse[(m, target)] = (rmse_map[(m, target)] - vmin) / denom

    # 综合评分
    scoring_rows = []
    for m in models_list:
        score = np.mean([norm_rmse[(m, t)] for t in OUTPUT_COLUMNS])
        scoring_rows.append({
            "model": m,
            "normalized_RMSE_y1": round(norm_rmse[(m, "thermal_resistance")], 4),
            "normalized_RMSE_y2": round(norm_rmse[(m, "pressure_drop")], 4),
            "normalized_RMSE_y3": round(norm_rmse[(m, "temperature_nonuniformity")], 4),
            "overall_score": round(score, 4),
            "RMSE_y1_raw": round(rmse_map[(m, "thermal_resistance")], 6),
            "RMSE_y2_raw": round(rmse_map[(m, "pressure_drop")], 6),
            "RMSE_y3_raw": round(rmse_map[(m, "temperature_nonuniformity")], 6),
            "R2_y1_mean": round(
                cv_df[(cv_df["model"] == m) & (cv_df["target"] == "thermal_resistance")]["R2_mean"].values[0], 4),
            "R2_y2_mean": round(
                cv_df[(cv_df["model"] == m) & (cv_df["target"] == "pressure_drop")]["R2_mean"].values[0], 4),
            "R2_y3_mean": round(
                cv_df[(cv_df["model"] == m) & (cv_df["target"] == "temperature_nonuniformity")]["R2_mean"].values[0], 4),
        })

    ranking_df = pd.DataFrame(scoring_rows).sort_values("overall_score")
    ranking_df = ranking_df.reset_index(drop=True)

    # 分配推荐角色
    best_model = ranking_df.iloc[0]["model"]
    roles = []
    for _, row in ranking_df.iterrows():
        m = row["model"]
        if m == best_model:
            role = "primary_surrogate"
        elif m == "RSM-2":
            role = "interpretability_baseline"
        elif m in ["GPR", "RSM-3-Ridge"]:
            role = "secondary_surrogate"
        else:
            role = "comparison"
        roles.append(role)
    ranking_df["recommended_role"] = roles

    # 保存
    path = os.path.join(OUTPUT_TABLES, "table_04_model_overall_ranking.csv")
    ranking_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[ModelSelection] Ranking saved -> table_04_model_overall_ranking.csv")

    # 输出摘要
    print("\n[ModelSelection] Overall Ranking:")
    for _, row in ranking_df.iterrows():
        print(f"  {row['model']:14s}  score={row['overall_score']:.4f}  "
              f"R2_y1={row['R2_y1_mean']:.4f}  R2_y2={row['R2_y2_mean']:.4f}  "
              f"R2_y3={row['R2_y3_mean']:.4f}  role={row['recommended_role']}")

    return {
        "ranking": ranking_df,
        "best_overall": best_model,
    }


def select_best_models(cv_df: pd.DataFrame, ranking: dict, models: dict) -> dict:
    """
    根据 CV 结果选择每个目标变量的最佳模型。
    采用 per-target 最优 + overall 最优比对策略。
    返回 {target: (model_name, pipeline)}
    """
    best_per_target = {}
    for target in OUTPUT_COLUMNS:
        sub = cv_df[cv_df["target"] == target].sort_values("RMSE_mean")
        best_model = sub.iloc[0]["model"]
        best_rmse = sub.iloc[0]["RMSE_mean"]
        best_r2 = sub.iloc[0]["R2_mean"]
        print(f"  [{OUTPUT_NAMES_CN[target]}] Best per-target: {best_model} "
              f"(RMSE={best_rmse:.6f}, R2={best_r2:.4f})")
        best_per_target[target] = (best_model, models[best_model])

    # 检查是否可以统一使用 overall best
    best_overall = ranking["best_overall"]
    print(f"\n  Best overall: {best_overall}")

    # 保存选择元信息
    summary = {
        "best_overall": best_overall,
        "best_per_target": {
            t: best_per_target[t][0] for t in OUTPUT_COLUMNS
        },
        "ranking": ranking["ranking"].to_dict(orient="records"),
    }
    ensure_dir(OUTPUT_MODELS)
    with open(MODEL_SELECTION_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Selection summary saved -> model_selection_summary.json")

    return best_per_target
