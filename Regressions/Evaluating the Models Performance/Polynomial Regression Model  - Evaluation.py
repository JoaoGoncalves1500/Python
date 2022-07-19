# Polynomial Regression Model Evaluation

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
data = pd.read_csv('Data.csv')
X = data.iloc[:,:-1].values # Independent Variables - 4 Features 
y = data.iloc[:,-1].values  # Dependent Variable


# Spliting the training set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Training the Polynomial Regression Model on the training set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly,y_train)


# Predicting the test set results
y_pred = regressor.predict(poly_reg.transform(X_test)) 
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)