import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.cluster import KMeans

data = datasets.load_iris()
iris = pd.DataFrame(data.data)
iris.columns = data.feature_names
print(iris.head())

print(data.target_names)
print(data.feature_names)

x = iris
print(x)

x_train, x_test = train_test_split(x)
model = KMeans(n_clusters=3)
pred = model.fit(x)
model.fit(x_train)
print(model)
