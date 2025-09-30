import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # feature (2D array)
y = np.array([2, 4, 6, 8, 10])  # target

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Get slope (coefficient) and intercept
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Predict values
y_pred = model.predict(X)

# Plot results
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
