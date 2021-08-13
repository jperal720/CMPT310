import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import IPython.display as display
import PIL.Image
import pickle
import random
import os
import cv2
import SimpleITK as sitk
import pandas
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from os import listdir
from numpy import asarray
from numpy import save
from numpy import load
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import cross_val_score

imgSize = 200
images = load('S:/Documents/CMPT310/test/dogs_vs_cats_images.npy', allow_pickle=True)
labels = load('S:/Documents/CMPT310/test/dogs_vs_cats_labels.npy', allow_pickle=True)

X = []
y = []

for image in images:
    X.append(image)

for label in labels:
    y.append(label)

# print(X.shape)
X = np.array(X).reshape(-1, imgSize, imgSize, 3)
y = np.array(y).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

#SVM Classifier

clf = svm.SVC()

# print(images.shape, labels.shape)
num_samples = y_train.shape[0]
num_samples_X = y.shape[0]
X_train = np.reshape(X_train, (num_samples, -1))
X = np.reshape(X, (num_samples_X, -1))
print(y.shape)

print(X_train.shape, y_train.shape)
print(X.shape, y.shape)
# clf.fit(X_train, y_train)


# #Prediction
#
y_prediction = clf.predict(clf, X_test)
# print(y_prediction)
print(clf.score(X_train, y_train))

