import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm

cancer = datasets.load_breast_cancer()
print(cancer)

data = pd.DataFrame(cancer.data)
print(data)

print(data.info())
print(cancer.feature_names)

data.columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                'mean smoothness', 'mean compactness', 'mean concavity',
                'mean concave points', 'mean symmetry', 'mean fractal dimension',
                'radius error', 'texture error', 'perimeter error', 'area error',
                'smoothness error', 'compactness error', 'concavity error',
                'concave points error', 'symmetry error',
                'fractal dimension error', 'worst radius', 'worst texture',
                'worst perimeter', 'worst area', 'worst smoothness',
                'worst compactness', 'worst concavity', 'worst concave points',
                'worst symmetry', 'worst fractal dimension']

x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(accuracy_score(y_test, y_predict) * 100)
