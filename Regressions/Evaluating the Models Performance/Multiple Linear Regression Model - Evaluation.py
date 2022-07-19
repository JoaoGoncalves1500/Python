# Multiple Linear Regression Model Evaluation

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


# Training the Multiple Linear Regression Model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)


# Predicting the test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
Prediction = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
Prediction[:5,:]


# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
