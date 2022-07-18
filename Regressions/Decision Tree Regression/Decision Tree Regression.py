# Decision Tree Regression

# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Import the dataset 
dataset = pd.read_csv('Position_Salaries.csv')
dataset

X = dataset.iloc[:,1:-1].values 
y = dataset.iloc[:,-1].values


# Training the Decision Tree Regression Model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)


# Visualising the Decision Tree Regression Model result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.scatter([[6.5]],regressor.predict([[6.5]]), color = 'yellow') # Predicting for X = 6.5
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
regressor.predict([[6.5]])

