"""
experiment_random.py -

Author:黄成钰
Date:2026/9/27
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 项目根目录
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# =========================
# 导入感知机
# =========================

from src.perceptron_original import Perceptron


# =========================
# 1. 生成线性可分数据
# =========================

np.random.seed(42)

n_samples = 100

# 第一类
X_positive = np.random.randn(n_samples // 2, 2) + np.array([2, 2])

# 第二类
X_negative = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])

X = np.vstack([
    X_positive,
    X_negative
])

y = np.hstack([
    np.ones(n_samples // 2),
    -np.ones(n_samples // 2)
])


# =========================
# 2. 训练感知机
# =========================

model = Perceptron()

model.fit(X, y)

print("===== Random Dataset =====")
print("w =", model.w)
print("b =", model.b)
print("epochs =", model.n_epochs)
print("updates =", model.n_updates)

# =========================
# 3. 计算准确率
# =========================

prediction = model.predict(X)

accuracy = np.mean(prediction == y)

print("accuracy =", accuracy)


# =========================
# 4. 绘制数据
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Positive (+1)"
)

plt.scatter(
    X[y == -1, 0],
    X[y == -1, 1],
    label="Negative (-1)"
)


# =========================
# 5. 绘制决策边界
# =========================

w = model.w
b = model.b

x1 = np.linspace(
    X[:, 0].min() - 1,
    X[:, 0].max() + 1,
    200
)

if abs(w[1]) > 1e-12:

    x2 = -(w[0] * x1 + b) / w[1]

    plt.plot(
        x1,
        x2,
        linestyle="--",
        linewidth=2,
        label="Decision Boundary"
    )


# =========================
# 6. 图像设置
# =========================

plt.xlabel("x1")
plt.ylabel("x2")

plt.title("Perceptron on Random Linearly Separable Data")

plt.legend()
plt.grid(True)

plt.tight_layout()


# =========================
# 7. 保存图片
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_PATH = FIGURE_DIR / "random_dataset.png"

plt.savefig(
    FIGURE_PATH,
    dpi=300
)

print("figure saved to:", FIGURE_PATH)

plt.show()