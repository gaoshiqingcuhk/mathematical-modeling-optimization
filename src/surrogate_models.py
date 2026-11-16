"""
代理模型构建模块
支持 5 种模型：RSM-2, RSM-3-Ridge, GPR, RF, GBDT
所有模型使用 sklearn Pipeline 确保无数据泄漏
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RepeatedKFold, cross_validate, LeaveOneOut
import joblib
import json
import warnings
import time
import os

from src.config import (
    INPUT_COLUMNS, OUTPUT_COLUMNS, OUTPUT_NAMES_CN,
    CV_RANDOM_SEED, CV_N_SPLITS, CV_N_REPEATS,
    MODEL_Y1, MODEL_Y2, MODEL_Y3, MODEL_SELECTION_JSON,
    OUTPUT_MODELS, OUTPUT_TABLES,
)
from src.utils import ensure_dir

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ── scoring 定义 ───────────────────────────────────────────
SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
}


def build_all_models() -> dict:
    """
    构建所有候选模型的字典。
    每个模型是一个未训练的 sklearn Pipeline (或兼容对象)。
    返回 {model_name: Pipeline}
    """
    models = {}

    # ═══════════════════════════════════════════════════════
    #  Model 1: RSM-2 (二次响应面 + Ridge)
    # ═══════════════════════════════════════════════════════
    models["RSM-2"] = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=CV_RANDOM_SEED)),
    ])

    # ═══════════════════════════════════════════════════════
    #  Model 2: RSM-3-Ridge (三次响应面 + RidgeCV)
    # ═══════════════════════════════════════════════════════
    models["RSM-3-Ridge"] = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
    ])

    # ═══════════════════════════════════════════════════════
    #  Model 3: GPR (Gaussian Process Regression)
    # ═══════════════════════════════════════════════════════
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
    models["GPR"] = Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=CV_RANDOM_SEED,
            normalize_y=False,
        )),
    ])

    # ═══════════════════════════════════════════════════════
    #  Model 4: Random Forest
    # ═══════════════════════════════════════════════════════
    models["RF"] = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=CV_RANDOM_SEED,
            n_jobs=1,
        )),
    ])

    # ═══════════════════════════════════════════════════════
    #  Model 5: Gradient Boosting
    # ═══════════════════════════════════════════════════════
    models["GBDT"] = Pipeline([
        ("scaler", StandardScaler()),
        ("gbdt", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=CV_RANDOM_SEED,
        )),
    ])

    return models


# ────────────────────────────────────────────────────────────
#  Repeated KFold 交叉验证
# ────────────────────────────────────────────────────────────
def run_repeated_kfold_cv(models: dict, X: np.ndarray, y_dict: dict) -> pd.DataFrame:
    """
    对每个模型 × 每个目标变量执行 RepeatedKFold CV。
    返回汇总 DataFrame。
    """
    cv = RepeatedKFold(
        n_splits=CV_N_SPLITS,
        n_repeats=CV_N_REPEATS,
        random_state=CV_RANDOM_SEED,
    )

    rows = []
    for model_name, pipe in models.items():
        for target in OUTPUT_COLUMNS:
            y = y_dict[target]
            t0 = time.time()
            scores = cross_validate(
                pipe, X, y,
                cv=cv,
                scoring=SCORING,
                n_jobs=1,
                return_train_score=False,
            )
            elapsed = time.time() - t0

            rmse = -scores["test_RMSE"]
            mae = -scores["test_MAE"]
            r2 = scores["test_R2"]

            rows.append({
                "model": model_name,
                "target": target,
                "target_cn": OUTPUT_NAMES_CN[target],
                "RMSE_mean": np.mean(rmse),
                "RMSE_std": np.std(rmse),
                "MAE_mean": np.mean(mae),
                "MAE_std": np.std(mae),
                "R2_mean": np.mean(r2),
                "R2_std": np.std(r2),
                "time_sec": round(elapsed, 1),
            })

            print(f"  [{model_name:12s} | {OUTPUT_NAMES_CN[target]:8s}] "
                  f"RMSE={np.mean(rmse):.6f}+/-{np.std(rmse):.6f}  "
                  f"MAE={np.mean(mae):.6f}+/-{np.std(mae):.6f}  "
                  f"R2={np.mean(r2):.4f}+/-{np.std(r2):.4f}  "
                  f"({elapsed:.1f}s)")

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
#  LOOCV（辅助验证）
# ────────────────────────────────────────────────────────────
def run_loocv(models: dict, X: np.ndarray, y_dict: dict,
              selected_models=None) -> pd.DataFrame:
    """
    Leave-One-Out CV，默认只对 RSM-2, RSM-3-Ridge, GPR 执行。
    """
    if selected_models is None:
        selected_models = ["RSM-2", "RSM-3-Ridge", "GPR"]

    loo = LeaveOneOut()
    rows = []

    for model_name in selected_models:
        if model_name not in models:
            continue
        pipe = models[model_name]
        for target in OUTPUT_COLUMNS:
            y = y_dict[target]
            t0 = time.time()
            scores = cross_validate(
                pipe, X, y,
                cv=loo,
                scoring=SCORING,
                n_jobs=1,
                return_train_score=False,
            )
            elapsed = time.time() - t0

            rmse = -scores["test_RMSE"]
            mae = -scores["test_MAE"]
            r2 = scores["test_R2"]

            rows.append({
                "model": model_name,
                "target": target,
                "target_cn": OUTPUT_NAMES_CN[target],
                "RMSE": np.mean(rmse),
                "MAE": np.mean(mae),
                "R2": np.mean(r2),
                "time_sec": round(elapsed, 1),
            })

            print(f"  [LOOCV {model_name:12s} | {OUTPUT_NAMES_CN[target]:8s}] "
                  f"RMSE={np.mean(rmse):.6f}  MAE={np.mean(mae):.6f}  "
                  f"R2={np.mean(r2):.4f}  ({elapsed:.1f}s)")

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
#  训练最佳模型并保存
# ────────────────────────────────────────────────────────────
def train_and_save_best_models(
    selected_models: dict,
    X: np.ndarray,
    y_dict: dict,
    df: pd.DataFrame,
) -> dict:
    """
    对每个输出指标，在全量数据上训练选定的最佳模型，并保存。
    selected_models: {target: (model_name, pipeline)}
    返回 {target: (model_name, fitted_pipeline)}
    注意：每个 target 使用独立的 clone，避免同模型族覆盖。
    """
    from sklearn.base import clone
    ensure_dir(OUTPUT_MODELS)
    trained = {}

    for target in OUTPUT_COLUMNS:
        model_name, pipe_template = selected_models[target]
        pipe = clone(pipe_template)  # 独立副本，避免 shared-object bug
        y = y_dict[target]
        pipe.fit(X, y)

        path_map = {
            "thermal_resistance": MODEL_Y1,
            "pressure_drop": MODEL_Y2,
            "temperature_nonuniformity": MODEL_Y3,
        }
        path = path_map[target]
        joblib.dump(pipe, path)
        print(f"  [{target}] {model_name} saved -> {os.path.basename(path)}")
        trained[target] = (model_name, pipe)

    # 生成全量预测
    rows = []
    for i in range(len(df)):
        x_i = X[i:i+1]
        row = {"sample_id": int(df.iloc[i]["sample_id"])}
        for j, target in enumerate(OUTPUT_COLUMNS):
            _, pipe = trained[target]
            y_true = y_dict[target][i]
            y_pred = float(pipe.predict(x_i)[0])
            row[f"true_{target}"] = y_true
            row[f"pred_{target}"] = y_pred
            row[f"residual_{target}"] = y_true - y_pred
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    path_pred = os.path.join(OUTPUT_TABLES, "table_05_best_model_predictions.csv")
    pred_df.to_csv(path_pred, index=False, encoding="utf-8-sig")
    print(f"  Predictions saved -> table_05_best_model_predictions.csv")

    return trained


# ────────────────────────────────────────────────────────────
#  提取 RSM-2 系数
# ────────────────────────────────────────────────────────────
def extract_rsm2_coefficients(rsm2_pipe, X: np.ndarray, y_dict: dict) -> pd.DataFrame:
    """
    提取二次响应面模型（标准化的）系数及其项名称，用于可解释性分析。
    """
    rows = []
    for target in OUTPUT_COLUMNS:
        # 重新训练 RSM-2 获取系数
        pipe_copy = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=CV_RANDOM_SEED)),
        ])
        pipe_copy.fit(X, y_dict[target])
        feature_names_raw = pipe_copy.named_steps["poly"].get_feature_names_out(INPUT_COLUMNS)
        coef = pipe_copy.named_steps["ridge"].coef_
        scaler = pipe_copy.named_steps["scaler"]
        # 将系数还原为原始尺度近似
        # beta_orig = coef / scaler.scale_
        beta_orig = coef / scaler.scale_

        for feat, c, c_orig in zip(feature_names_raw, coef, beta_orig):
            rows.append({
                "target": target,
                "target_cn": OUTPUT_NAMES_CN[target],
                "term": feat,
                "coefficient_std": round(float(c), 8),
                "coefficient_orig": round(float(c_orig), 8),
                "abs_coefficient_std": round(abs(float(c)), 8),
                "sign": "+" if c > 0 else "-",
            })

    df_coef = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_TABLES, "table_06_rsm2_coefficients.csv")
    df_coef.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  RSM-2 coefficients saved -> table_06_rsm2_coefficients.csv")
    return df_coef
