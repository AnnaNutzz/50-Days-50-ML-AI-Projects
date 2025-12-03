import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([8, 10, 12], dtype=float)
y = np.array([10, 13, 16], dtype=float)

class LinearRegressionGD:
    def __init__(self, lr=0.001, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.w = 0
        self.b = 0

    def predict(self, X):
        return self.w * X + self.b

    def fit(self, X, y):
        n = len(X)
        losses = []

        for epoch in range(self.epochs):
            y_pred = self.predict(X)

            # Compute gradients
            dw = (2/n) * np.sum((y_pred - y) * X)
            db = (2/n) * np.sum(y_pred - y)

            # Update weights
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Track loss
            loss = np.mean((y_pred - y)**2)
            losses.append(loss)

        return losses

model = LinearRegressionGD(lr=0.001, epochs=3000)
losses = model.fit(X, y)

print("GD Slope (w):", model.w)
print("GD Intercept (b):", model.b)

# Plot the loss curve
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Convergence")
plt.show()
