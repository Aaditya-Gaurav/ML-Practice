import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example dataset (hours studied vs exam result)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)  # hours studied
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])  # pass(1)/fail(0)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y, y_pred))

# Probability estimates
y_prob = model.predict_proba(X)[:, 1]  # probability of class 1
print("Probabilities:", y_prob)

# Plot
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_prob, color="red", label="Predicted probability")
plt.xlabel("Hours studied")
plt.ylabel("Probability of Passing")
plt.legend()
plt.show()
