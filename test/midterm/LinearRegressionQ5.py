import csv
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pandas as pd

###
# LINEAR REGRESSION MODEL
# Data Partition

x = {'Age': [20, 18, 25, 28, 33, 40, 45,], 'Height': [64, 60, 58, 63, 66, 54, 51]}

y = {'Annual Income': [50000, 38000, 25000, 10000, 55000, 50000, 250000]}

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)

# #Creating MLP classifier Model
# linReg = LinearRegression()
# pipeLine = make_pipeline(columnTrans, linReg)
# print("CVS:", cross_val_score(pipeLine, x, y, cv=3).mean())

# ### Prediction
# xSample = x.head(50) #grabs the first 50 rows of x
# ySample = y.head(50).to_numpy() #grabs the first 50 rows of y
# pipeLine.fit(x, y)
# yPred = pipeLine.predict(xSample)
# print(yPred)
# print(pipeLine.score(x, y))
# print("this is the new shape X", xSample)
# print("this is the new shape", yPred)

# ### Graph
# yPred.sort(axis=0)
# plt.plot(yPred, 'o', color='black')
# plt.plot(ySample, 'o', color='red')

# plt.show()