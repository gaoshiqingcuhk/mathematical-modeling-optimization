"""
全局配置：随机种子、列名、路径等
"""
import os

# ── 随机种子 ────────────────────────────────────────────
RANDOM_SEED = 42

# ── 项目根目录 ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 数据路径 ────────────────────────────────────────────
RAW_DATA = os.path.join(BASE_DIR, "data", "raw", "附件2_性能数据.xlsx")
CLEAN_DATA = os.path.join(BASE_DIR, "data", "processed", "clean_data.csv")
NORMALIZED_DATA = os.path.join(BASE_DIR, "data", "processed", "normalized_data.csv")

# ── 输出文件路径 ────────────────────────────────────────
OUTPUT_MODELS = os.path.join(BASE_DIR, "outputs", "models")
OUTPUT_TABLES = os.path.join(BASE_DIR, "outputs", "tables")
OUTPUT_FIGURES = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_LOGS = os.path.join(BASE_DIR, "outputs", "logs")

# ── 输入变量列名 ────────────────────────────────────────
INPUT_COLUMNS = [
    "pin_width_ratio",
    "manifold_depth_height_ratio",
    "pin_rows",
]

# ── 输出变量列名 ────────────────────────────────────────
OUTPUT_COLUMNS = [
    "thermal_resistance",
    "pressure_drop",
    "temperature_nonuniformity",
]

# ── 中文名称映射 ────────────────────────────────────────
INPUT_NAMES_CN = {
    "pin_width_ratio": "针肋宽度比",
    "manifold_depth_height_ratio": "歧管深高比",
    "pin_rows": "针肋排数",
}

OUTPUT_NAMES_CN = {
    "thermal_resistance": "无量纲热阻",
    "pressure_drop": "无量纲压降",
    "temperature_nonuniformity": "无量纲温度非均匀性",
}

# ── 变量取值范围 ────────────────────────────────────────
X_BOUNDS = {
    "pin_width_ratio": (0.0, 0.3),
    "manifold_depth_height_ratio": (3.0, 4.5),
    "pin_rows": (0, 10),
}

# ── 优化方向（全为 min） ─────────────────────────────────
OPT_DIRECTION = {
    "thermal_resistance": "min",
    "pressure_drop": "min",
    "temperature_nonuniformity": "min",
}

# ── 交叉验证配置 ────────────────────────────────────────
CV_RANDOM_SEED = 2026
CV_N_SPLITS = 5
CV_N_REPEATS = 20

# ── 模型文件路径 ────────────────────────────────────────
MODEL_Y1 = os.path.join(OUTPUT_MODELS, "best_model_y1.joblib")
MODEL_Y2 = os.path.join(OUTPUT_MODELS, "best_model_y2.joblib")
MODEL_Y3 = os.path.join(OUTPUT_MODELS, "best_model_y3.joblib")
MODEL_SELECTION_JSON = os.path.join(OUTPUT_MODELS, "model_selection_summary.json")

# ── 绘图配置 ────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免 GUI 依赖

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

# 尝试中文字体，若不可用则使用英文
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
