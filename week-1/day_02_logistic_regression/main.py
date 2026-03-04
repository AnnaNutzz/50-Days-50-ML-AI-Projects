"""
Day 02 — Logistic Regression (from scratch)
Binary classification using gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt


X = np.array([2, 3, 4, 5, 6, 7], dtype=float)
y = np.array([0, 0, 0, 1, 1, 1], dtype=float)


a0 = 0.0
a1 = 0.0
lr = 0.1
epochs = 5000


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for _ in range(epochs):
    z = a0 + a1 * X
    y_pred = sigmoid(z)

    da0 = np.mean(y_pred - y)
    da1 = np.mean((y_pred - y) * X)

    a0 -= lr * da0
    a1 -= lr * da1


print("Learned intercept (a0):", round(a0, 3))
print("Learned coefficient (a1):", round(a1, 3))


x_curve = np.linspace(0, 8, 200)
y_curve = sigmoid(a0 + a1 * x_curve)


plt.figure(figsize=(8, 5))
plt.plot(x_curve, y_curve, linewidth=2, label="Sigmoid Curve")
plt.scatter(X, y, color="black", zorder=3, label="Training Data")

plt.xlabel("Hours Studied")
plt.ylabel("Probability of PASS")
plt.title("Logistic Regression Sigmoid Curve")
plt.legend()
plt.grid(True)
plt.show()


decision_boundary = -a0 / a1


plt.figure(figsize=(8, 5))
plt.plot(x_curve, y_curve, linewidth=2, label="Sigmoid Curve")
plt.scatter(X, y, color="black", zorder=3, label="Training Data")

plt.axvline(
    x=decision_boundary,
    linestyle="--",
    label=f"Decision Boundary (x = {decision_boundary:.2f})"
)

plt.xlabel("Hours Studied")
plt.ylabel("Probability of PASS")
plt.title("Sigmoid Curve with Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
