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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from os import listdir
from numpy import asarray
from numpy import save
from tensorflow import keras

dataDir = "S:/Documents/CMPT310/test/PetImages"
categories = ["Dog", "Cat"]

trainingData = []
imgSize = 200

(train_dataset, test_dataset), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

# location of downloaded Dogs vs. Cats image dataset on my computer
# folder = '/Volumes/WDBook/dogs-vs-cats/train/'

# subset of original dataset consisting of 1,000 images
folder = '/Volumes/WDBook/dogs-vs-cats/train-1000/'

# subset of original dataset consisting of 10,000 images
# folder = '/Volumes/WDBook/dogs-vs-cats/train-10000/'

resized_photo = load_img("/path/to/original_image_file", target_size=(200, 200))