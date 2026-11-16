#!/usr/bin/env python
"""
APMCM B题 一键复现脚本
用法: python run_all.py
"""
import sys
import os

# 确保 src 可导入
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.config import RANDOM_SEED
from src.data_loader import load_and_clean, load_and_normalize
from src.eda import run_eda
from src.plotting import plot_correlation_heatmap
from src.utils import ensure_dir


def phase1():
    """第一阶段：数据读取、清洗、EDA"""
    print("\n" + "=" * 60)
    print("Phase 1: Data Loading, Cleaning, EDA")
    print("=" * 60)

    df = load_and_clean()
    df_norm = load_and_normalize(df)
    eda_results = run_eda(df)
    plot_correlation_heatmap(df)

    return df, df_norm, eda_results


def phase2(df):
    """第二阶段：代理模型训练、交叉验证、模型选择"""
    import numpy as np
    import pandas as pd
    from src.config import INPUT_COLUMNS, OUTPUT_COLUMNS, CV_RANDOM_SEED
    from src.surrogate_models import (
        build_all_models,
        run_repeated_kfold_cv,
        run_loocv,
        train_and_save_best_models,
        extract_rsm2_coefficients,
    )
    from src.model_selection import rank_models, select_best_models
    from src.plot_modeling import (
        plot_model_comparison_rmse,
        plot_pred_vs_true,
        plot_residuals_best_model,
        plot_rsm2_coefficient_importance,
    )

    print("\n" + "=" * 60)
    print("Phase 2: Surrogate Model Training & CV")
    print("=" * 60)

    # 准备数据
    X = df[INPUT_COLUMNS].values.astype(float)
    y_dict = {col: df[col].values.astype(float) for col in OUTPUT_COLUMNS}

    # 1. 构建模型
    models = build_all_models()
    print(f"\n[Phase2] Total models: {len(models)}")
    for name, pipe in models.items():
        print(f"  {name}: {[s[0] for s in pipe.steps]}")

    # 2. Repeated KFold CV
    print(f"\n[Phase2] Running Repeated KFold CV (n_splits=5, n_repeats=20, seed={CV_RANDOM_SEED})...")
    cv_df = run_repeated_kfold_cv(models, X, y_dict)

    # 保存 CV 指标表
    from src.config import OUTPUT_TABLES
    import os
    cv_path = os.path.join(OUTPUT_TABLES, "table_03_model_cv_metrics.csv")
    cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Phase2] CV metrics saved -> table_03_model_cv_metrics.csv")

    # 3. LOOCV (selected models)
    print(f"\n[Phase2] Running LOOCV (RSM-2, RSM-3-Ridge, GPR)...")
    loocv_df = run_loocv(models, X, y_dict, selected_models=["RSM-2", "RSM-3-Ridge", "GPR"])
    loocv_path = os.path.join(OUTPUT_TABLES, "table_07_loocv_metrics.csv")
    loocv_df.to_csv(loocv_path, index=False, encoding="utf-8-sig")
    print(f"[Phase2] LOOCV metrics saved -> table_07_loocv_metrics.csv")

    # 4. 模型排序与选择
    print(f"\n[Phase2] Model ranking & selection...")
    ranking = rank_models(cv_df)
    best_per_target = select_best_models(cv_df, ranking, models)

    # 5. 在全量数据上训练最佳模型
    print(f"\n[Phase2] Training best models on full data...")
    trained_models = train_and_save_best_models(best_per_target, X, y_dict, df)

    # 6. RSM-2 系数提取
    print(f"\n[Phase2] Extracting RSM-2 coefficients...")
    rsm2_pipe = models["RSM-2"]
    coef_df = extract_rsm2_coefficients(rsm2_pipe, X, y_dict)

    # 7. 生成图表
    print(f"\n[Phase2] Generating Phase 2 figures...")
    plot_model_comparison_rmse(cv_df)
    plot_pred_vs_true(trained_models, X, y_dict)
    plot_residuals_best_model(trained_models, X, y_dict)
    plot_rsm2_coefficient_importance(coef_df)

    print("\n[Phase2] Done!")
    return {
        "cv_df": cv_df,
        "loocv_df": loocv_df,
        "ranking": ranking,
        "best_per_target": best_per_target,
        "trained_models": trained_models,
        "rsm2_coefficients": coef_df,
        "models": models,
    }


def phase3(df, p2_results):
    """Phase 3: Multi-objective optimization, Pareto, TOPSIS"""
    import numpy as np
    from src.config import OUTPUT_COLUMNS, INPUT_COLUMNS
    from src.optimization import (
        sample_space_optimization,
        surrogate_grid_search,
        build_candidate_comparison,
        unified_gpr_check,
    )
    from src.plot_optimization import (
        plot_pareto_2d_projections,
        plot_pareto_3d,
        plot_sample_space_top10,
        plot_final_candidates_comparison,
    )

    print("\n" + "=" * 60)
    print("Phase 3: Multi-Objective Optimization")
    print("=" * 60)

    # 加载主代理模型
    trained_models = p2_results["trained_models"]

    # ── 3.1 样本空间优化 ────────────────────────────
    sample_results = sample_space_optimization(df)

    # ── 3.2 代理模型网格搜索 ─────────────────────────
    surrogate_results = surrogate_grid_search(df, trained_models)

    # ── 3.3 候选方案对比 ────────────────────────────
    comp_df = build_candidate_comparison(df, sample_results, surrogate_results, trained_models)

    # ── 3.4 统一 GPR 复核 ────────────────────────────
    gpr_check = unified_gpr_check(df, surrogate_results)

    # ── 3.5 绘图 ─────────────────────────────────────
    # 样本空间 Pareto
    res_df = sample_results["result"]
    Y_sample = res_df[OUTPUT_COLUMNS].values
    pareto_mask = sample_results["pareto_mask"]
    Y_pareto_sample = Y_sample[pareto_mask]

    key_samples = {}
    for sid in [64, 48, 13, 80]:
        row = df[df["sample_id"] == sid].iloc[0]
        key_samples[f"#{sid}"] = (row["thermal_resistance"], row["pressure_drop"],
                                   row["temperature_nonuniformity"])

    plot_pareto_2d_projections(Y_sample, pareto_mask, Y_pareto_sample, key_samples)
    plot_pareto_3d(Y_sample, pareto_mask, Y_pareto_sample, key_samples)

    # Top 10 bar charts
    plot_sample_space_top10(sample_results)

    # Final candidates comparison
    plot_final_candidates_comparison(comp_df)

    print("\n[Phase3] Done!")

    return {
        "sample_results": sample_results,
        "surrogate_results": surrogate_results,
        "comp_df": comp_df,
        "gpr_check": gpr_check,
    }


def phase4(df, p3_results):
    """Phase 4: Weight robustness analysis & robust design selection"""
    from src.robustness import run_phase4
    from src.plot_robustness import plot_all_phase4

    print("\n" + "=" * 60)
    print("Phase 4: Weight Robustness & Robust Design")
    print("=" * 60)

    p4_results = run_phase4(df, p3_results)
    plot_all_phase4(p4_results)

    print("\n[Phase4] Done!")
    return p4_results


def phase5(df=None):
    """Phase 5: Manufacturing & operational sensitivity"""
    from src.sensitivity import run_phase5
    from src.plot_sensitivity import plot_all_phase5

    print("\n" + "=" * 60)
    print("Phase 5: Manufacturing & Operational Sensitivity")
    print("=" * 60)

    p5_results = run_phase5(df)
    plot_all_phase5(p5_results)

    print("\n[Phase5] Done!")
    return p5_results


def main():
    print("=" * 60)
    print("APMCM B: Chip Thermal Management Optimization")
    print(f"Random Seed (EDA): {RANDOM_SEED}")
    print("=" * 60)

    # ── 创建必要目录 ──
    from src.config import OUTPUT_LOGS, OUTPUT_TABLES, OUTPUT_FIGURES, OUTPUT_MODELS
    from src.config import CLEAN_DATA, NORMALIZED_DATA
    for d in [OUTPUT_LOGS, OUTPUT_TABLES, OUTPUT_FIGURES, OUTPUT_MODELS]:
        ensure_dir(d)
    ensure_dir(os.path.dirname(CLEAN_DATA))

    # ══════════════════════════════════════════════════════
    #  Phase 1
    # ══════════════════════════════════════════════════════
    df, df_norm, eda_results = phase1()

    # ══════════════════════════════════════════════════════
    #  Phase 2
    # ══════════════════════════════════════════════════════
    p2_results = phase2(df)

    # ══════════════════════════════════════════════════════
    #  Phase 3
    # ══════════════════════════════════════════════════════
    p3_results = phase3(df, p2_results)

    # ══════════════════════════════════════════════════════
    #  Phase 4
    # ══════════════════════════════════════════════════════
    p4_results = phase4(df, p3_results)

    # ══════════════════════════════════════════════════════
    #  Phase 5
    # ══════════════════════════════════════════════════════
    p5_results = phase5(df)

    print("\n" + "=" * 60)
    print("[DONE] Phase 1, 2, 3, 4 & 5 Completed!")
    print(f"  Tables: {OUTPUT_TABLES}")
    print(f"  Figures: {OUTPUT_FIGURES}")
    print(f"  Models: {OUTPUT_MODELS}")
    print("=" * 60)

    return df, df_norm, eda_results, p2_results, p3_results, p4_results, p5_results


if __name__ == "__main__":
    main()
