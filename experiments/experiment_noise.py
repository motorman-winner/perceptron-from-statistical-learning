"""
experiment_noise.py -

Author:黄成钰
Date:2026/9/27
"""
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
# 4. 实验参数
# =========================

n_train = 500
n_test = 5000

noise_levels = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20
]


# 保存结果

updates_list = []
epochs_list = []
train_accuracy_list = []
test_accuracy_list = []


# =========================
# 5. 生成测试集
# =========================

X_test = np.random.uniform(
    -2,
    2,
    size=(n_test, 2)
)

# 真实标签

y_test = np.where(
    X_test[:, 0] + X_test[:, 1] > 0,
    1,
    -1
)


# =========================
# 6. 不同噪声水平实验
# =========================

for noise in noise_levels:

    # -------------------------
    # 生成训练数据
    # -------------------------

    X_train = np.random.uniform(
        -2,
        2,
        size=(n_train, 2)
    )

    # 真实标签

    y_train = np.where(
        X_train[:, 0] + X_train[:, 1] > 0,
        1,
        -1
    )

    # -------------------------
    # 添加标签噪声
    # -------------------------

    n_noise = int(
        n_train * noise
    )

    noise_indices = np.random.choice(
        n_train,
        size=n_noise,
        replace=False
    )

    y_train[noise_indices] *= -1

    # -------------------------
    # 创建模型
    # -------------------------

    model = Perceptron()

    # -------------------------
    # 训练
    # -------------------------

    model.fit(
        X_train,
        y_train,
        max_epochs=1000
    )

    # -------------------------
    # 训练集预测
    # -------------------------

    train_prediction = model.predict(
        X_train
    )

    train_accuracy = np.mean(
        train_prediction == y_train
    )

    # -------------------------
    # 测试集预测
    # -------------------------

    test_prediction = model.predict(
        X_test
    )

    test_accuracy = np.mean(
        test_prediction == y_test
    )

    # -------------------------
    # 保存结果
    # -------------------------

    updates_list.append(
        model.n_updates
    )

    epochs_list.append(
        model.n_epochs
    )

    train_accuracy_list.append(
        train_accuracy
    )

    test_accuracy_list.append(
        test_accuracy
    )

    # -------------------------
    # 输出
    # -------------------------

    print(
        f"noise = {noise:.2f} | "
        f"epochs = {model.n_epochs:4d} | "
        f"updates = {model.n_updates:6d} | "
        f"train_acc = {train_accuracy:.4f} | "
        f"test_acc = {test_accuracy:.4f}"
    )


# =========================
# 7. 绘制测试准确率
# =========================

plt.figure(figsize=(8, 6))

plt.plot(
    noise_levels,
    test_accuracy_list,
    marker="o",
    linewidth=2
)

plt.xlabel("Noise Level")

plt.ylabel("Test Accuracy")

plt.title(
    "Perceptron Test Accuracy vs Label Noise"
)

plt.grid(True)

plt.tight_layout()


# =========================
# 8. 保存图片
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
    / "accuracy_vs_noise.png"
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