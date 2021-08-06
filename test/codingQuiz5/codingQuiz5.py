import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.datasets import make_regression

# Create synthetic regression dataset with 1,000 examples
# Each example has 5 input variables
# The dataset has 3 features that are important and 2 features that are redundant
# random_state=1 creates reproducible output across multiple function calls
#     https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
X, y = make_regression(n_samples=1000, n_features=5, n_informative=3, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

training_data = pd.DataFrame(X_train, columns=['a', 'b', 'c', 'd', 'e'])

training_data.insert(0, "Y", y_train)

sns.pairplot(training_data, kind='reg', diag_kind='kde')
