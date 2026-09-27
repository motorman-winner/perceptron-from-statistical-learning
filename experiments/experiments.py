from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.perceptron_original import Perceptron
from src.perceptron_dual import PerceptronDual


# =========================
# 1. 构造数据
# =========================

X = np.array([
    [3, 3],
    [4, 3],
    [1, 1],
    [2, 1]
])

y = np.array([
    1,
    1,
    -1,
    -1
])


# =========================
# 2. 原始形式
# =========================

original = Perceptron()

original.fit(X, y)

print("===== Original Perceptron =====")
print("w =", original.w)
print("b =", original.b)

original_pred = original.predict(X)

print("prediction =", original_pred)
print("true label =", y)


# =========================
# 3. 对偶形式
# =========================

dual = PerceptronDual()

dual.fit(X, y)

print("\n===== Dual Perceptron =====")
print("alpha =", dual.alpha)
print("w =", dual.w)
print("b =", dual.b)

dual_pred = dual.predict(X)

print("prediction =", dual_pred)
print("true label =", y)


# =========================
# 4. 可视化
# =========================

plt.figure(figsize=(8, 6))

# 正样本
plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    marker="o",
    s=100,
    label="Positive (+1)"
)

# 负样本
plt.scatter(
    X[y == -1, 0],
    X[y == -1, 1],
    marker="x",
    s=100,
    label="Negative (-1)"
)


# =========================
# 5. 画决策边界
# =========================

w = original.w
b = original.b

x1 = np.linspace(0, 5, 100)

# w1*x1 + w2*x2 + b = 0
#
# x2 = -(w1*x1 + b) / w2

if w[1] != 0:

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

plt.title("Perceptron Decision Boundary")

plt.xlim(0, 5)
plt.ylim(0, 5)

plt.legend()
plt.grid(True)

plt.tight_layout()


# =========================
# 7. 保存图片
# =========================

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 创建结果文件夹
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 保存图片
FIGURE_PATH = FIGURE_DIR / "perceptron_boundary.png"

plt.savefig(
    FIGURE_PATH,
    dpi=300
)

print("figure saved to:", FIGURE_PATH)

plt.show()
