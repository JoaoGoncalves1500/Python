# Random Forest Regression


# Importing the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


# Training the Random Forest Regression Model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X,y)


# Visualising the Random Forest Regression Model results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = 'Red')
plt.plot(X_grid,regressor.predict(X_grid), color = 'blue')
plt.scatter([[6.5]],regressor.predict([[6.5]]), color = 'yellow')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
regressor.predict([[6.5]])