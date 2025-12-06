import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
#importing the dataset
dataset = pd.read_csv(r'D:\New folder\position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values #we require just numerical data 
y = dataset.iloc[:, -1].values

#training the linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)

#training the linear regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
#visualizing the linear regression result
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or bluff(Linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualizing the polynomial regression result
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue') # it contains power
plt.title('Truth or bluff(Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#visualizing the polynomial regression result(higher resolution and smooth curve)
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue') # it contains power
plt.title('Truth or bluff(Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#predicting a new result in linear regreesion(at a particular point)
regressor.predict([[6.5]])
#predicting a new result in polynomial regreesion(at a particular point)
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
