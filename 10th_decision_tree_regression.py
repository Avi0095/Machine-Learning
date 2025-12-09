import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset = pd.read_csv(r'D:\New folder\position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values #we require just numerical data 
y = dataset.iloc[:, -1].values

#training the decision tree regression on whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#predicting a new result
regressor.predict([[6.5]])
#visualize the result on higher resolution
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Decision tree regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()