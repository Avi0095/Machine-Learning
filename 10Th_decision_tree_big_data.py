import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv(r'D:\New folder\decision_tree_big_dataset.csv')

x = dataset.iloc[:, :-1].values   # 3 features
y = dataset.iloc[:, -1].values    # target

# Train Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

# Predict example
print(regressor.predict([[60.1115, 42.02942, 34.30516]]))

# -----------------------------
# Get means of other features
# -----------------------------
f1 = x[:,0]      # Feature1
f2 = x[:,1]      # Feature2
f3 = x[:,2]      # Feature3

f1_mean = np.mean(f1)
f2_mean = np.mean(f2)
f3_mean = np.mean(f3)

# -----------------------------
# Plot 1: Feature1 vs Prediction
# -----------------------------
x1_grid = np.arange(min(f1), max(f1), 0.1).reshape(-1, 1)

x1_full = np.column_stack((x1_grid,
                           np.full_like(x1_grid, f2_mean),
                           np.full_like(x1_grid, f3_mean)))

plt.figure(figsize=(7,5))
plt.scatter(f1, y, color='red')
plt.plot(x1_grid, regressor.predict(x1_full), color='blue')
plt.title('Decision Tree Regression — Feature1')
plt.xlabel('Feature1')
plt.ylabel('Target')
plt.grid(True)
plt.show()

# -----------------------------
# Plot 2: Feature2 vs Prediction
# -----------------------------
x2_grid = np.arange(min(f2), max(f2), 0.5).reshape(-1, 1)

x2_full = np.column_stack((np.full_like(x2_grid, f1_mean),
                           x2_grid,
                           np.full_like(x2_grid, f3_mean)))

plt.figure(figsize=(7,5))
plt.scatter(f2, y, color='red')
plt.plot(x2_grid, regressor.predict(x2_full), color='green')
plt.title('Decision Tree Regression — Feature2')
plt.xlabel('Feature2')
plt.ylabel('Target')
plt.grid(True)
plt.show()

# -----------------------------
# Plot 3: Feature3 vs Prediction
# -----------------------------
x3_grid = np.arange(min(f3), max(f3), 0.1).reshape(-1, 1)

x3_full = np.column_stack((np.full_like(x3_grid, f1_mean),
                           np.full_like(x3_grid, f2_mean),
                           x3_grid))

plt.figure(figsize=(7,5))
plt.scatter(f3, y, color='red')
plt.plot(x3_grid, regressor.predict(x3_full), color='purple')
plt.title('Decision Tree Regression — Feature3')
plt.xlabel('Feature3')
plt.ylabel('Target')
plt.grid(True)
plt.show()