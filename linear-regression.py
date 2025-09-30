import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Mean of X and y
X_mean = np.mean(X)
y_mean = np.mean(y)

# Calculating slope (m) and intercept (b)
num = np.sum((X - X_mean) * (y - y_mean))
den = np.sum((X - X_mean) ** 2)
m = num / den
b = y_mean - m * X_mean

print(f"Slope (m): {m}, Intercept (b): {b}")

# Predictions
y_pred = m * X + b

# Plotting
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
