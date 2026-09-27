"""
experiment_train_test.py -

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
# 3. 设置随机种子
# =========================

np.random.seed(42)


# =========================
# 4. 不同训练集规模
# =========================

sample_sizes = [20, 50, 100, 200, 500, 1000]


# 保存实验结果
epochs_list = []
updates_list = []
train_accuracy_list = []
test_accuracy_list = []


# =========================
# 5. 固定测试集
# =========================

n_test = 2000

X_test = np.random.uniform(
    -2,
    2,
    size=(n_test, 2)
)

y_test = np.where(
    X_test[:, 0] + X_test[:, 1] > 0,
    1,
    -1
)


# =========================
# 6. 依次改变训练集大小
# =========================

for n_samples in sample_sizes:

    # -------------------------
    # 生成训练集
    # -------------------------

    X_train = np.random.uniform(
        -2,
        2,
        size=(n_samples, 2)
    )

    y_train = np.where(
        X_train[:, 0] + X_train[:, 1] > 0,
        1,
        -1
    )

    # -------------------------
    # 创建模型
    # -------------------------

    model = Perceptron()

    # -------------------------
    # 训练
    # -------------------------

    model.fit(
        X_train,
        y_train
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

    epochs_list.append(
        model.n_epochs
    )

    updates_list.append(
        model.n_updates
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
        f"n = {n_samples:4d} | "
        f"epochs = {model.n_epochs:4d} | "
        f"updates = {model.n_updates:4d} | "
        f"train_acc = {train_accuracy:.4f} | "
        f"test_acc = {test_accuracy:.4f}"
    )


# =========================
# 7. 绘制训练集 / 测试集准确率
# =========================

plt.figure(figsize=(8, 6))

plt.plot(
    sample_sizes,
    train_accuracy_list,
    marker="o",
    linewidth=2,
    label="Train Accuracy"
)

plt.plot(
    sample_sizes,
    test_accuracy_list,
    marker="s",
    linewidth=2,
    label="Test Accuracy"
)

plt.xlabel("Number of Training Samples")

plt.ylabel("Accuracy")

plt.title(
    "Perceptron Training and Test Accuracy"
)

plt.ylim(
    0.8,
    1.01
)

plt.legend()

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
    / "train_test_accuracy.png"
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
