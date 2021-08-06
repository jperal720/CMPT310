import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2, f_regression


# ###SECTION 1 (Training 1000 samples and plotting them in seaborn)
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

# ###SECTION 2 (Mean Squared Error)
#
# y_true = [3, -0.5, 2, 7]
# y_pred = [2.5, 0.0, 2, 8]
#
# print(mean_squared_error(y_true, y_pred))
#
# y_true = [3, -0.5, 2, 7]
# y_pred = [2.5, 0.0, 2, 8]
#
# print(mean_squared_error(y_true, y_pred, squared=False))
#
# y_true = [[0.5, 1],[-1, 1],[7, -6]]
# y_pred = [[0, 2],[-1, 2],[8, -5]]
#
# print(mean_squared_error(y_true, y_pred))
#
# print(mean_squared_error(y_true, y_pred, squared=False))

# ###Section 3 (Mean Absolute Error)
#
# y_true = [3, -0.5, 2, 7]
# y_pred = [2.5, 0.0, 2, 8]
#
# print(mean_absolute_error(y_true, y_pred))
#
# y_true = [[0.5, 1], [-1, 1], [7, -6]]
# y_pred = [[0, 2], [-1, 2], [8, -5]]
#
# print(mean_absolute_error(y_true, y_pred))

# ###Section 4
#
# model_with_3_best_features = SelectKBest(score_func=f_regression, k=3) #To construct model that selects '3' as its best features
#
# model_with_all_features = SelectKBest(score_func=f_regression, k='all') #To construct model that selects 'all' as its best features

###Example
# Create a synthetic regression dataset consisting of 100 examples
#     where each example has 10 input features (with 10 of the features being informative)
X, y = make_regression(n_samples=100, n_features=10, n_informative=10, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Set up the model to select its 5 best features
model_with_5_best_features = SelectKBest(score_func=f_regression, k=5)

# train the model
model_with_5_best_features.fit(X_train, y_train)

# transform preserves the K best features in the train input data and discards all the other features
X_train_5_best_features = model_with_5_best_features.transform(X_train)

# transform preserves the K best features in the test input data and discards all the other features
X_test_5_best_features = model_with_5_best_features.transform(X_test)

# scores for each of the 10 features
for i in range(len(model_with_5_best_features.scores_)):
	print('Feature %d: %f' % (i, model_with_5_best_features.scores_[i]))

plt.bar([i for i in range(len(model_with_5_best_features.scores_))], model_with_5_best_features.scores_)

plt.show()

###Pair Plot of '5 best features' data

training_data = pd.DataFrame(X_train_5_best_features, columns=['a', 'b', 'c', 'd', 'e'])

training_data.insert(0, 'Y', y_train)

sns.pairplot(training_data, kind='reg', diag_kind='kde')

plt.show()
