"""
experiments -

Author:黄成钰
Date:2026/9/26
"""
import numpy as np

from src.perceptron_original import Perceptron


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


model = Perceptron()

model.fit(X, y)

print("w =", model.w)
print("b =", model.b)

pred = model.predict(X)

print("prediction =", pred)
print("true label =", y)