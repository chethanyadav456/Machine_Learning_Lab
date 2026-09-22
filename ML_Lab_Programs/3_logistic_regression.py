import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

data = load_iris()
iris_data = pd.DataFrame(data.data)
iris_data.columns = data.feature_names
print(iris_data.head())

x = iris_data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(accuracy_score(y_test, y_predict) * 100)
