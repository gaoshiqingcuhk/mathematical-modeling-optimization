"""
表格生成与保存
"""
import pandas as pd
import numpy as np
import os
from src.config import (
    OUTPUT_TABLES, INPUT_COLUMNS, OUTPUT_COLUMNS,
    INPUT_NAMES_CN, OUTPUT_NAMES_CN,
)
from src.utils import ensure_dir, weighted_sum_score, topsis_score


def _ensure_table_dir():
    ensure_dir(OUTPUT_TABLES)


def save_data_description(df: pd.DataFrame):
    """table_01: 数据描述统计"""
    _ensure_table_dir()

    rows = []
    for col in INPUT_COLUMNS + OUTPUT_COLUMNS:
        name_cn = INPUT_NAMES_CN.get(col) or OUTPUT_NAMES_CN.get(col) or col
        series = df[col]
        rows.append({
            "变量": col,
            "中文名称": name_cn,
            "样本数": len(series),
            "最小值": series.min(),
            "最大值": series.max(),
            "均值": series.mean(),
            "标准差": series.std(),
            "缺失数": series.isnull().sum(),
            "数据类型": str(series.dtype),
        })

    table = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_TABLES, "table_01_data_description.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Tables] 已保存: table_01_data_description.csv")


def save_single_objective_optima(optima: dict):
    """table_02: 单目标最优表"""
    _ensure_table_dir()

    rows = []
    for col in OUTPUT_COLUMNS:
        d = optima[col]
        rows.append({
            "优化目标": OUTPUT_NAMES_CN[col],
            "变量名": col,
            "样本编号": d["sample_id"],
            "pin_width_ratio": d["pin_width_ratio"],
            "manifold_depth_height_ratio": d["manifold_depth_height_ratio"],
            "pin_rows": d["pin_rows"],
            "thermal_resistance": d["thermal_resistance"],
            "pressure_drop": d["pressure_drop"],
            "temperature_nonuniformity": d["temperature_nonuniformity"],
        })

    table = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_TABLES, "table_02_single_objective_optima.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Tables] 已保存: table_02_single_objective_optima.csv")


def save_equal_weight_top10(
    df: pd.DataFrame,
    top10_ws_idx: np.ndarray,
    top10_topsis_idx: np.ndarray,
    Z: np.ndarray,
    w_eq: np.ndarray,
):
    """保存等权综合最优 Top 10（加权和 + TOPSIS）"""
    _ensure_table_dir()

    # 一次性计算所有样本的 TOPSIS 和加权和
    all_ws = weighted_sum_score(Z, w_eq)       # (n_samples,)
    all_topsis = topsis_score(Z, w_eq)          # (n_samples,)

    # 加权和 Top 10
    rows_ws = []
    for rank, idx in enumerate(top10_ws_idx, start=1):
        row = df.iloc[idx]
        rows_ws.append({
            "排名": rank,
            "样本编号": int(row["sample_id"]),
            "pin_width_ratio": row["pin_width_ratio"],
            "manifold_depth_height_ratio": row["manifold_depth_height_ratio"],
            "pin_rows": int(row["pin_rows"]),
            "thermal_resistance": row["thermal_resistance"],
            "pressure_drop": row["pressure_drop"],
            "temperature_nonuniformity": row["temperature_nonuniformity"],
            "加权和得分": round(float(all_ws[idx]), 6),
            "TOPSIS贴近度": round(float(all_topsis[idx]), 6),
        })

    path_ws = os.path.join(OUTPUT_TABLES, "table_02b_equal_weight_top10_ws.csv")
    pd.DataFrame(rows_ws).to_csv(path_ws, index=False, encoding="utf-8-sig")
    print(f"[Tables] 已保存: table_02b_equal_weight_top10_ws.csv")

    # TOPSIS Top 10
    rows_topsis = []
    for rank, idx in enumerate(top10_topsis_idx, start=1):
        row = df.iloc[idx]
        rows_topsis.append({
            "排名": rank,
            "样本编号": int(row["sample_id"]),
            "pin_width_ratio": row["pin_width_ratio"],
            "manifold_depth_height_ratio": row["manifold_depth_height_ratio"],
            "pin_rows": int(row["pin_rows"]),
            "thermal_resistance": row["thermal_resistance"],
            "pressure_drop": row["pressure_drop"],
            "temperature_nonuniformity": row["temperature_nonuniformity"],
            "加权和得分": round(float(all_ws[idx]), 6),
            "TOPSIS贴近度": round(float(all_topsis[idx]), 6),
        })

    path_topsis = os.path.join(OUTPUT_TABLES, "table_02c_equal_weight_top10_topsis.csv")
    pd.DataFrame(rows_topsis).to_csv(path_topsis, index=False, encoding="utf-8-sig")
    print(f"[Tables] 已保存: table_02c_equal_weight_top10_topsis.csv")
