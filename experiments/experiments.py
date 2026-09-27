import numpy as np

from src.perceptron_original import Perceptron
from src.perceptron_dual import PerceptronDual


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
# 原始形式
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
# 对偶形式
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