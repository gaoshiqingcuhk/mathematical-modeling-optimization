"""
工具函数：归一化、距离计算、随机数生成器等
"""
import numpy as np
import pandas as pd
from src.config import RANDOM_SEED

# 全局随机数生成器
rng = np.random.default_rng(RANDOM_SEED)


def minmax_normalize(y: np.ndarray) -> np.ndarray:
    """
    Min-max 归一化，输出范围 [0, 1]。
    对越小越好的指标，归一化后仍然越小越好。
    """
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)
    denom = y_max - y_min
    denom[denom == 0] = 1e-12
    return (y - y_min) / denom


def weighted_sum_score(Z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    归一化后的加权和，假设 Z 越小越好。
    Z: (n_samples, n_objectives)
    weights: (n_objectives,) 或 (n_weight_sets, n_objectives)
    """
    if weights.ndim == 1:
        return Z @ weights
    else:
        return Z @ weights.T  # (n_samples, n_weight_sets)


def topsis_score(Z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    计算 TOPSIS 贴近度。
    Z: (n_samples, n_objectives) 归一化后越小越好
    weights: (n_objectives,)
    返回贴近度 C，越大越好。
    """
    weights = np.asarray(weights, dtype=float)
    ideal = np.zeros(Z.shape[1])
    anti_ideal = np.ones(Z.shape[1])

    # Paper-facing TOPSIS convention: fixed ideal=(0,0,0),
    # anti-ideal=(1,1,1), with all objectives lower-is-better.
    D_plus = np.sqrt(np.sum(weights * (Z - ideal) ** 2, axis=1))
    D_minus = np.sqrt(np.sum(weights * (Z - anti_ideal) ** 2, axis=1))

    denom = D_plus + D_minus
    denom[denom == 0] = 1e-12
    C = D_minus / denom
    return C


def ensure_dir(path: str):
    """确保目录存在"""
    import os
    os.makedirs(path, exist_ok=True)


def format_float(x, decimals=6):
    """统一浮点数格式化"""
    return round(float(x), decimals)
