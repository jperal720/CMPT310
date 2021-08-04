import csv
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pandas as pd

###
# LINEAR REGRESSION MODEL
# Data Partition

xInches = pd.DataFrame(data={'Age': [20, 18, 25, 28, 33, 40, 45], 'Height': [64, 60, 58, 63, 66, 54, 51]})
xFeet = pd.DataFrame(data={'Age': [20, 18, 25, 28, 33, 40, 45], 'Height': [64 / 12, 60 / 12, 58 / 12, 63 / 12, 66 / 12, 54 / 12, 51 / 12]})

y = pd.DataFrame(data={'Annual Income': [50000, 38000, 25000, 10000, 55000, 50000, 250000]})

X_train, X_test, Y_train, Y_test = train_test_split(xInches, y, train_size=0.8)

print('Linear Regression (inches)')

# Creating Linear Regression Model (inches)
linReg = LinearRegression()
linReg.fit(X_train, Y_train)

### Prediction
yPred = linReg.predict(X_test)


### Results
print('Coefficients:', linReg.coef_)
print('Intercept', linReg.intercept_)
print('MSE: %.2f' % mean_squared_error(Y_test, yPred))
print('Coefficient of determination (R^2): %.2f' % r2_score(Y_test, yPred))

# yPred.sort(axis=0)
plt.plot(yPred, 'o', color='black')
plt.plot(Y_test, 'o', color='red')

plt.show()

print()
print('Linear Regression (feet)')

# Creating Linear Regression Model (inches)
linReg = LinearRegression()
linReg.fit(X_train, Y_train)


#Linear Regression (feet)
X_train, X_test, Y_train, Y_test, = train_test_split(xFeet, y, train_size=0.8)

linReg = LinearRegression()
linReg.fit(X_train, Y_train)

### Prediction
yPred = linReg.predict(X_test)


### Results
print('Coefficients:', linReg.coef_)
print('Intercept', linReg.intercept_)
print('MSE: %.2f' % mean_squared_error(Y_test, yPred))
print('Coefficient of determination (R^2): %.2f' % r2_score(Y_test, yPred))

# yPred.sort(axis=0)
plt.plot(yPred, 'o', color='black')
plt.plot(Y_test, 'o', color='red')

plt.show()
# Linear Regression Model (feet)
