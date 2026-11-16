"""
EDA: 描述统计、单目标最优点、等权综合最优点、主效应
"""
import pandas as pd
import numpy as np
from src.config import INPUT_COLUMNS, OUTPUT_COLUMNS, INPUT_NAMES_CN, OUTPUT_NAMES_CN
from src.utils import minmax_normalize, weighted_sum_score, topsis_score
from src.tables import (
    save_data_description,
    save_single_objective_optima,
    save_equal_weight_top10,
)
from src.plotting import plot_main_effects


def run_eda(df: pd.DataFrame):
    """
    执行完整的 EDA 流程：
    1. 输出数据描述统计表
    2. 找出单目标最优点
    3. 等权综合最优 Top 10
    4. 生成主效应图
    """
    print("\n" + "=" * 60)
    print("[EDA] 开始探索性数据分析")
    print("=" * 60)

    # ── 1. 数据描述统计表 ─────────────────────────────────
    save_data_description(df)

    # ── 2. 单目标最优点 ───────────────────────────────────
    single_objective_optima = {}
    for col in OUTPUT_COLUMNS:
        idx = df[col].idxmin()
        row = df.loc[idx]
        single_objective_optima[col] = {
            "sample_id": int(row["sample_id"]),
            "pin_width_ratio": row["pin_width_ratio"],
            "manifold_depth_height_ratio": row["manifold_depth_height_ratio"],
            "pin_rows": int(row["pin_rows"]),
            "thermal_resistance": row["thermal_resistance"],
            "pressure_drop": row["pressure_drop"],
            "temperature_nonuniformity": row["temperature_nonuniformity"],
        }
        print(f"[EDA] 最小 {OUTPUT_NAMES_CN[col]}: 样本 {int(row['sample_id'])}, "
              f"x=({row['pin_width_ratio']}, {row['manifold_depth_height_ratio']}, {int(row['pin_rows'])}), "
              f"值={row[col]:.6f}")

    save_single_objective_optima(single_objective_optima)

    # ── 3. 等权综合最优前 10 ──────────────────────────────
    # Paper-facing equal-weight ranking follows the Phase 3 candidate set:
    # enhanced pin-fin designs only, excluding no-pin baselines.
    df_enhanced = df[(df["pin_width_ratio"] > 0) & (df["pin_rows"] > 0)].copy()
    df_enhanced = df_enhanced.reset_index(drop=True)
    y_raw = df_enhanced[OUTPUT_COLUMNS].values
    Z = minmax_normalize(y_raw)
    w_eq = np.array([1 / 3, 1 / 3, 1 / 3])

    # 等权加权和
    scores_ws = weighted_sum_score(Z, w_eq)
    df_enhanced["ws_score"] = scores_ws

    # TOPSIS 贴近度
    C = topsis_score(Z, w_eq)
    df_enhanced["topsis_score"] = C

    # 加权和排名 (越小越好)
    top10_ws_idx = np.argsort(scores_ws)[:10]
    top10_ws = df_enhanced.iloc[top10_ws_idx].copy()
    top10_ws["ws_rank"] = range(1, 11)
    top10_ws = top10_ws.sort_values("ws_rank")

    print(f"\n[EDA] 等权综合最优 (加权和) Top 3:")
    for _, row in top10_ws.head(3).iterrows():
        print(f"  样本 {int(row['sample_id'])}, x=({row['pin_width_ratio']}, "
              f"{row['manifold_depth_height_ratio']}, {int(row['pin_rows'])}), "
              f"ws_score={row['ws_score']:.6f}, topsis={row['topsis_score']:.6f}")

    # TOPSIS 排名 (越大越好)
    top10_topsis_idx = np.argsort(C)[::-1][:10]

    save_equal_weight_top10(df_enhanced, top10_ws_idx, top10_topsis_idx, Z, w_eq)

    # ── 4. 主效应图 ───────────────────────────────────────
    plot_main_effects(df)

    print("[EDA] EDA 完成")
    print("=" * 60 + "\n")

    return {
        "single_objective_optima": single_objective_optima,
        "top10_ws": top10_ws,
        "top10_topsis_idx": top10_topsis_idx,
    }
