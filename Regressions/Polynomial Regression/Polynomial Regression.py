# Polynomial Regression Model

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
dataset
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


# Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


# Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # Change the degree to increase the accuracy of the prediction
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


# Linear Regression Results
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title('Linear Regression Result')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Polynomial Regression Results
plt.scatter(X,y, color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Polynomail Regression Result')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Making a higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regression Result 2')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Prediction with the Linear Regression Model
np.set_printoptions(precision=2)
lin_reg.predict([[7]])

# Prediction with the Plynomial Regression Model
lin_reg_2.predict(poly_reg.fit_transform([[7]]))