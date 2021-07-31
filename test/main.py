import csv
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

studentMath = pd.read_csv('student-mat.csv', ';')
print(studentMath.shape)
print(studentMath.isna().sum())

###
# Ignore 4, 11, 20 and keeps the other 30 rows
#
studentMath = studentMath.loc[:, ['school', 'sex', 'age', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'G1', 'G2', 'G3'] ]

print(studentMath.shape)

# with open ('student-mat.csv', 'r') as csvFile:
#     csvReader = csv.reader(csvFile)
#
#     for row in csvReader:
#         studentMath.append(row)
#
#     print(studentMath)
