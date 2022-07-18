# Data Processing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
dataset=pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # Features (Indepedent variables) - iloc - indeces locate
y = dataset.iloc[:, -1].values  # Depedent variable 

print(X)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])
print(X)

# Encoding categorical data

# Encoding the Independent variable
# This serves to give each country a number and seperate them into different columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Features scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train[:, 3:] = ss.fit_transform(X_train[:, 3:])
X_test[:, 3:] = ss.transform(X_test[:, 3:])
print(X_train)
print(X_test)