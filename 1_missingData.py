import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Load dataset
dataset = pd.read_csv(r'D:\New folder\dataset_with_missing.csv')

# Independent and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Original X:")
print(X)
print(y)

# Handle missing values (Age and Salary)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print("\nX after imputation:")
print(X)