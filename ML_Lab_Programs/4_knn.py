import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

data = datasets.load_iris()
iris_data = pd.DataFrame(data.data)
iris_data.columns = data.feature_names
print(iris_data.head())

x = iris_data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = KNeighborsClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(accuracy_score(y_test, y_predict) * 100)
print(confusion_matrix(y_test, y_predict))

data1 = datasets.load_breast_cancer()
print(data1)

cancer = pd.DataFrame(data1.data)
cancer.columns = data1.feature_names
print(cancer)

x1 = cancer.iloc[:, 1:]
y1 = cancer.iloc[:, 0]
print(y1)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1)
model1 = KNeighborsRegressor()
model1.fit(x1_train, y1_train)

y1_Predict = model1.predict(x1_test)
print(r2_score(y1_test, y1_Predict) * 100)
