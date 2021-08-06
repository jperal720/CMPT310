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
# MultiLayerPerceptron
# Data Partition
# Ignore 1, 2, 4, 5, 11, 12, 20, 23, G1, G2 and keeps the other rows

studentMath = pd.read_csv('student-mat.csv', ';')
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

#ONE HOT ENCODER
ohe = OneHotEncoder(sparse=False)
# print(ohe.fit_transform(x[['Pstatus']]))
# print(ohe.categories_)

columnTrans = make_column_transformer((OneHotEncoder(),
                                      ['Mjob', 'Fjob', 'schoolsup', 'famsup', 'paid', 'activities',
                                       'higher', 'internet']), remainder='passthrough')

#Creating MLP classifier Model
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
pipeLine = make_pipeline(columnTrans, clf)
print("CVS:", cross_val_score(pipeLine, x, y, cv=3).mean())

### Prediction

xSample = x.head(50) #grabs the first 50 rows of x
ySample = y.head(50).to_numpy() #grabs the first 50 rows of y
pipeLine.fit(x, y)
yPred = pipeLine.predict(xSample)
print(yPred)
print(pipeLine.score(x, y))
print("this is the new shape X", xSample)
print("this is the new shape", yPred)

### Graph
# xSample = columnTrans.fit_transform(xSample)
# print(xSample)
yPred.sort(axis=0)
plt.plot(yPred, 'o', color='black')
plt.plot(ySample, 'o', color='red')

plt.show()
