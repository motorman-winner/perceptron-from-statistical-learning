"""
perceptron_original -

Author:黄成钰
Date:2026/9/26
"""
import numpy as np

class Perceptron:

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        while True:

            error_count = 0

            for i in range(n_samples):

                if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
                    self.w += y[i] * X[i]
                    self.b += y[i]

                    error_count += 1

            if error_count == 0:
                break

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


