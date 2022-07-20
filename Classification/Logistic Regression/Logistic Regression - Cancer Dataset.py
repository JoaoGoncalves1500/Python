# Logistic Regression Model on the cancer dataset

# import the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
data = pd.read_csv('breast_cancer.csv')
data
X = data.iloc[:,1:-1].values # excluded the first column ('Sample code number')
y = data.iloc[:, -1].values  # The dependent variable ('classe')- 2 for benign, 4 for malignant


# Finding values which are empty strings
np.where(data.applymap(lambda x: x == ''))

# Finding the row and column indices where the value is NaN
np.where(pd.isnull(data))


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Computing the accuracy with k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f}%".format(accuracies.std()*100))