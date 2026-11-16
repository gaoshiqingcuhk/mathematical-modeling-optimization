"""
数据读取与清洗
"""
import pandas as pd
import numpy as np
from src.config import RAW_DATA, CLEAN_DATA, NORMALIZED_DATA, INPUT_COLUMNS, OUTPUT_COLUMNS, X_BOUNDS
from src.utils import ensure_dir, minmax_normalize
import os


def load_and_clean() -> pd.DataFrame:
    """
    读取附件2 Excel，清洗并保存。
    返回干净的 DataFrame。
    """
    # 读取：第0行为标题，第1行为字段名
    df = pd.read_excel(RAW_DATA, header=1)

    # 重命名字段
    df.columns = [
        "sample_id",
        "pin_width_ratio",
        "manifold_depth_height_ratio",
        "pin_rows",
        "thermal_resistance",
        "pressure_drop",
        "temperature_nonuniformity",
    ]

    # 类型转换
    df["sample_id"] = df["sample_id"].astype(int)
    df["pin_width_ratio"] = df["pin_width_ratio"].astype(float)
    df["manifold_depth_height_ratio"] = df["manifold_depth_height_ratio"].astype(float)
    df["pin_rows"] = df["pin_rows"].astype(int)
    df["thermal_resistance"] = df["thermal_resistance"].astype(float)
    df["pressure_drop"] = df["pressure_drop"].astype(float)
    df["temperature_nonuniformity"] = df["temperature_nonuniformity"].astype(float)

    # ── 数据质量检查 ──────────────────────────────────────
    n_samples = len(df)
    n_missing = df.isnull().sum().sum()
    n_duplicated = df.duplicated().sum()

    print(f"[DataLoader] 样本数: {n_samples}")
    print(f"[DataLoader] 缺失值总数: {n_missing}")
    print(f"[DataLoader] 重复行数: {n_duplicated}")

    # 检查变量取值范围
    for col in INPUT_COLUMNS:
        lo, hi = X_BOUNDS[col]
        actual_lo = df[col].min()
        actual_hi = df[col].max()
        if actual_lo < lo or actual_hi > hi:
            print(f"  [WARN] {col}: actual [{actual_lo}, {actual_hi}] exceeds nominal [{lo}, {hi}]")
        else:
            print(f"  [OK] {col}: [{actual_lo}, {actual_hi}]")

    for col in OUTPUT_COLUMNS:
        print(f"  [OK] {col}: min={df[col].min():.6f}, max={df[col].max():.6f}, "
              f"mean={df[col].mean():.6f}, std={df[col].std():.6f}")

    # 保存清洗数据
    ensure_dir(os.path.dirname(CLEAN_DATA))
    df.to_csv(CLEAN_DATA, index=False)
    print(f"[DataLoader] 清洗数据已保存至: {CLEAN_DATA}")

    return df


def load_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    对输出指标做 min-max 归一化，保存并返回。
    """
    y_raw = df[OUTPUT_COLUMNS].values
    z = minmax_normalize(y_raw)

    df_norm = df.copy()
    df_norm["z_thermal_resistance"] = z[:, 0]
    df_norm["z_pressure_drop"] = z[:, 1]
    df_norm["z_temperature_nonuniformity"] = z[:, 2]

    ensure_dir(os.path.dirname(NORMALIZED_DATA))
    df_norm.to_csv(NORMALIZED_DATA, index=False)
    print(f"[DataLoader] 归一化数据已保存至: {NORMALIZED_DATA}")

    return df_norm
