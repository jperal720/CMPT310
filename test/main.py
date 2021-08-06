import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# ###SECTION 1
# # Create synthetic regression dataset with 1,000 examples
# # Each example has 5 input variables
# # The dataset has 3 features that are important and 2 features that are redundant
# # random_state=1 creates reproducible output across multiple function calls
# #     https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
# X, y = make_regression(n_samples=1000, n_features=5, n_informative=3, random_state=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#
# print(y_train)
#
# training_data = pd.DataFrame(X_train, columns=['a', 'b', 'c', 'd', 'e'])
#
# training_data.insert(0, "Y", y_train)
#
# sns.pairplot(training_data, kind='reg', diag_kind='kde')
#
# print(training_data)
#
# plt.show()

###SECTION 2

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

print(mean_squared_error(y_true, y_pred))

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

print(mean_squared_error(y_true, y_pred, squared=False))

y_true = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]

print(mean_squared_error(y_true, y_pred))

print(mean_squared_error(y_true, y_pred, squared=False))