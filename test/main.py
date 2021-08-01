import csv
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pandas as pd

studentMath = pd.read_csv('student-mat.csv', ';')
# print(studentMath.shape)
# print(studentMath.isna().sum())

###
# LINEAR REGRESSION MODEL
# Ignore 1, 2, 4, 5, 11, 12, 20, 23, G1, G2 and keeps the other rows
#
studentMath = studentMath.loc[:, ['school', 'age', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities',
       'higher', 'internet', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'G3']]

x = studentMath.loc[:, ['age', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities',
       'higher', 'internet', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences']]

y = studentMath.G3
# x = studentMath.loc[:, ['studytime']]

# print(x.shape)
print(x.head)
# print(y.shape)

#ONE HOT ENCODER
ohe = OneHotEncoder(sparse=False)
# print(ohe.fit_transform(x[['Pstatus']]))
# print(ohe.categories_)

columnTrans = make_column_transformer((OneHotEncoder(),
                                      ['Mjob', 'Fjob', 'schoolsup', 'famsup', 'paid', 'activities',
                                       'higher', 'internet']), remainder='passthrough')
print("columnTransform: ", columnTrans.fit_transform(x))

#Creating Linear Regression Model
linReg = LinearRegression()
pipeLine = make_pipeline(columnTrans, linReg)
print(cross_val_score(pipeLine, x, y, cv=3).mean())


# print("CVS:", cross_val_score(linReg, x, y, cv=3).mean())
# print(y.value_counts(normalize=True))
#
# yPred = linReg.predict(x)
#
# print('Coefficients: \n', linReg.coef_)
# print('Mean squared error: %.2f' % mean_squared_error(y, yPred))
# print('Coefficient of determination: %.2f' % r2_score(y, yPred))
#
#
# plt.scatter(x, y, edgecolors='red')
# plt.plot(x, yPred, 'o', color='black')
# plt.xticks(())
# plt.yticks(())
#
# plt.show()
