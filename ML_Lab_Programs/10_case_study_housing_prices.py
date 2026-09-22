"""
Case Study 2: Predicting Housing Prices

Scenario:
A real estate agency wants to predict housing prices based on different property features
such as square footage, number of bedrooms, and location. Accurate price predictions can
help agents set fair property values and assist buyers in making informed decisions.

Objective:
Develop a regression model to predict the price of a house using key property features.

Concept Used:
Supervised Learning (Regression) — the model learns from labeled data (features + target
price) and predicts continuous values such as house prices. We use Linear Regression,
which assumes a linear relationship between features (inputs) and price (output).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Square_Feet': [1000, 1500, 2000, 2500, 3000, 3500],
    'Bedrooms': [2, 3, 3, 4, 4, 5],
    'Price': [150000, 200000, 250000, 280000, 350000, 400000]
}

df = pd.DataFrame(data)
print("Housing Dataset:\n", df)

X = df[['Square_Feet', 'Bedrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

print("\nCoefficients:")
print("Square_Feet Coefficient:", model.coef_[0])
print("Bedrooms Coefficient:", model.coef_[1])
print("Intercept:", model.intercept_)

# Evaluation:
# - The R2 Score indicates the strength of the linear relationship between features
#   and price.
# - Coefficients show how much the price increases per additional square foot or bedroom.

# Conclusion:
# This case study demonstrates how Linear Regression can predict housing prices using
# features like square footage and number of bedrooms. The model successfully learns
# the relationship between input features and target price, showcasing a basic
# supervised learning regression approach in machine learning.
