import numpy as np
import matplotlib.pyplot as plt


# Dataset 

X = np.array([2, 3, 4, 5, 6, 7], dtype=float)
y = np.array([0, 0, 0, 1, 1, 1], dtype=float)


# Initialize parameters

a0 = 0.0   # intercept
a1 = 0.0   # coefficient
lr = 0.1
epochs = 5000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent

for _ in range(epochs):
    z = a0 + a1 * X
    y_pred = sigmoid(z)

    da0 = np.mean(y_pred - y)
    da1 = np.mean((y_pred - y) * X)

    a0 -= lr * da0
    a1 -= lr * da1

print("Learned a0 (intercept):", round(a0, 3))
print("Learned a1 (coefficient):", round(a1, 3))


# Generate smooth x values
x_curve = np.linspace(0, 8, 200)
y_curve = sigmoid(a0 + a1 * x_curve)

# Plot sigmoid curve
plt.figure(figsize=(8, 5))
plt.plot(x_curve, y_curve, label="Sigmoid Curve", linewidth=2)

# Plot original data points
plt.scatter(X, y, color="black", zorder=3, label="Training Data")

plt.xlabel("Hours Studied")
plt.ylabel("Probability of PASS")
plt.title("Logistic Regression Sigmoid Curve")
plt.legend()
plt.grid(True)
plt.show()

# Decision boundary
decision_boundary = -a0 / a1

plt.figure(figsize=(8, 5))
plt.plot(x_curve, y_curve, label="Sigmoid Curve", linewidth=2)
plt.scatter(X, y, color="black", zorder=3, label="Training Data")

# boundary line
plt.axvline(
    x=decision_boundary,
    color="red",
    linestyle="--",
    label=f"Decision Boundary (x = {decision_boundary:.2f})"
)

plt.xlabel("Hours Studied")
plt.ylabel("Probability of PASS")
plt.title("Sigmoid Curve with Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()
