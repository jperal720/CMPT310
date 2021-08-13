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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from os import listdir
from numpy import asarray
from numpy import save
from tensorflow import keras
from tensorflow.keras import layers

dataDir = "S:/Documents/CMPT310/test/PetImages"
categories = ["Dog", "Cat"]

trainingData = []
imgSize = 200

(train_ds, test_ds), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True,
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features['label'].num_classes
print(num_classes)

get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))

def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    iamge = tf.image.resize(image, [imgSize, imgSize])
    image = (image / 255.0)
    return image, label

def augment(image_label, seed):
    image, label = image_label
    image, label = resize_and_rescale(image, label)
    image = tf.image.resize_with_crop_or_pad(image, imgSize + 6, imgSize + 6)

    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    image = tf.image.stateless_random_crop(image, size=[imgSize, imgSize, 3], seed=seed)

    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)

    image = tf.clip_by_value(image, 0, 1)

    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

train_ds = train_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(metadata.splits["train[:80%]"].num_examples)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(AUTOTUNE)

test_ds = test_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(128)
test_ds = test_ds.prefetch(AUTOTUNE)


model = keras.Sequential([
    keras.Input((28, 28, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(10)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# model.fit(train_dataset, epochs=5, verbose=2)
# model.evaluate(train_dataset)
#
# print(metadata)


# for image in train_dataset:
#     for index in image:
#         print(index)
#     break
#     # print(image)
#     sArr = sitk.GetArrayViewFromImage(image)
#     plt.imshow(sArr)
# location of downloaded Dogs vs. Cats image dataset on my computer
# folder = '/Volumes/WDBook/dogs-vs-cats/train/'

# subset of original dataset consisting of 1,000 images
folder = 'C:\\Users\\terupuki\\tensorflow_datasets\\cats_vs_dogs\\4.0.0'

X = []
X = np.array(X).reshape(2,  imgSize, imgSize)

# subset of original dataset consisting of 10,000 images
# folder = '/Volumes/WDBook/dogs-vs-cats/train-10000/'

# resized_photo = load_img("/path/to/original_image_file", target_size=(200, 200))