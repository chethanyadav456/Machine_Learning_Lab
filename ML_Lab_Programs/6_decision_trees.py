import numpy as np
import pandas as pd
from sklearn import datasets
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import r2_score

data = datasets.load_breast_cancer()
cancer = pd.DataFrame(data.data)
print(cancer)

cancer.columns = data.feature_names
cancer["target"] = data["target"]
print(cancer)

print(cancer.isnull().any())

print(cancer.info())
print(cancer.corr().sum())

x = cancer.iloc[:, 0:10]
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(accuracy_score(y_test, y_predict) * 100)

print(confusion_matrix(y_test, y_predict) * 100)

model1 = DecisionTreeRegressor()
model1.fit(x_train, y_train)

y_predict = model1.predict(x_test)
print(accuracy_score(y_test, y_predict) * 100)

print(r2_score(y_test, y_predict) * 100)

print(confusion_matrix(y_test, y_predict) * 100)
