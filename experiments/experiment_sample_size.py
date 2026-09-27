import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 项目根目录
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# =========================
# 2. 导入感知机
# =========================

from src.perceptron_original import Perceptron


# =========================
# 3. 随机种子
# =========================

np.random.seed(42)


# =========================
# 4. 样本数量
# =========================

sample_sizes = [20, 50, 100, 200, 500, 1000]


epochs_list = []
updates_list = []
accuracy_list = []


# =========================
# 5. 进行实验
# =========================

for n_samples in sample_sizes:

    # -------------------------
    # 生成二维随机数据
    # -------------------------

    X = np.random.uniform(
        -2,
        2,
        size=(n_samples, 2)
    )

    # -------------------------
    # 根据真实边界 x1 + x2 = 0
    # 生成标签
    # -------------------------

    y = np.where(
        X[:, 0] + X[:, 1] > 0,
        1,
        -1
    )

    # -------------------------
    # 训练感知机
    # -------------------------

    model = Perceptron()

    model.fit(X, y)

    # -------------------------
    # 预测
    # -------------------------

    prediction = model.predict(X)

    accuracy = np.mean(
        prediction == y
    )

    # -------------------------
    # 保存结果
    # -------------------------

    epochs_list.append(
        model.n_epochs
    )

    updates_list.append(
        model.n_updates
    )

    accuracy_list.append(
        accuracy
    )

    # -------------------------
    # 输出
    # -------------------------

    print(
        f"n = {n_samples:4d} | "
        f"epochs = {model.n_epochs:4d} | "
        f"updates = {model.n_updates:4d} | "
        f"accuracy = {accuracy:.4f}"
    )


# =========================
# 6. 绘制更新次数
# =========================

plt.figure(figsize=(8, 6))

plt.plot(
    sample_sizes,
    updates_list,
    marker="o",
    linewidth=2
)

plt.xlabel("Number of Samples")

plt.ylabel("Number of Updates")

plt.title(
    "Perceptron Updates vs Sample Size"
)

plt.grid(True)

plt.tight_layout()


# =========================
# 7. 保存图片
# =========================

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_PATH = (
    FIGURE_DIR
    / "updates_vs_sample_size.png"
)

plt.savefig(
    FIGURE_PATH,
    dpi=300
)

print()
print(
    "figure saved to:",
    FIGURE_PATH
)

plt.show()