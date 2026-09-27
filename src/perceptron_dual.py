"""
perceptron_dual.py -

Author:黄成钰
Date:2026/9/26
"""
import numpy as np


class PerceptronDual:

    def __init__(self):
        self.alpha = None
        self.b = None
        self.w = None

    def fit(self, X, y):

        n_samples = X.shape[0]

        # 初始化 alpha 和 b
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Gram 矩阵
        # G[i][j] = x_i^T x_j
        gram = X @ X.T

        while True:

            error_count = 0

            for i in range(n_samples):

                # w = Σ(alpha_j * y_j * x_j)
                # 因此 w^T x_i
                # = Σ(alpha_j * y_j * x_j^T x_i)
                result = np.sum(
                    self.alpha * y * gram[:, i]
                ) + self.b

                # 判断是否误分类
                if y[i] * result <= 0:

                    # alpha_i += 1
                    self.alpha[i] += 1

                    # b += y_i
                    self.b += y[i]

                    error_count += 1

            # 一轮下来没有误分类
            if error_count == 0:
                break

        # 根据 alpha 恢复 w
        self.w = np.sum(
            (self.alpha * y)[:, None] * X,
            axis=0
        )

    def predict(self, X):

        return np.sign(
            np.dot(X, self.w) + self.b
        )
