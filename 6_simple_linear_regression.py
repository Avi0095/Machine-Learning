import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
dataset = pd.read_csv(r'D:\New folder\house_price_data.csv')

# Independent and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set result
y_pred=regressor.predict(X_test)

#visualizing the training set result
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experiance(Training set)')
plt.xlabel('Year of Experiance')
plt.ylabel('Salary')
plt.show()

#visualizing the test set result
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue') # same because it is the exact straight line for comparing 
plt.title('Salary vs Experiance(Test set)')
plt.xlabel('Year of Experiance')
plt.ylabel('Salary')
plt.show()
