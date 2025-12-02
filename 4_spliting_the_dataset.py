import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
print("\n After OneHotEncoding:")
print(X)
le=LabelEncoder()
y=le.fit_transform(y)
print("\n The dependent variable:")
print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
print("\n train x at 80%")
print(X_train)
print("\n test x at 20%")
print(X_test)
print("\n train y at 80%")
print(y_train)
print("\n train y at 20%")
print(y_test)