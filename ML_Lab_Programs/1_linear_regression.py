import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# USING DEFAULT DATASETS
reg_data = datasets.fetch_california_housing()
print(reg_data)

data = pd.DataFrame(reg_data.data)
target = reg_data.target
print(target)

print(data.shape)
print(data)
print(reg_data.feature_names)

data.columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                'Latitude', 'Longitude']
print(data.head())

print(data.corr().sum)

x = data.loc[:, ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                  'Latitude', 'Longitude']]
y = target
x_train, x_test, y_train, y_test = train_test_split(x, y)
print(data.info())

model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(r2_score(y_test, y_predict) * 100)
