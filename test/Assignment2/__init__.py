import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image
import numpy as np
import IPython.display as display
import PIL.Image
import pickle
import random
import os
import cv2

dataDir = "S:/Documents/CMPT310/test/PetImages"
categories = ["Dog", "Cat"]

trainingData = []
imgSize = 200

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'S:/Documents/CMPT310/test/PetImages',
    labels='inferred',
    label_mode="int", #categorical, binary
    color_mode='rgb',
    batch_size=2,
    image_size=(imgSize, imgSize),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
)


def resizeAndRescale(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [imgSize, imgSize])
    image = (image / 255.0)
    print("debug")

    return image

def createTrainingData():
    for category in categories:
        path = os.path.join(dataDir, category)
        classNum = categories.index(category)
        num = 0
        print("hello")

        for image in os.listdir(path):
            try:
                if(num >= 20):
                    break
                num += 1
                print(num)
                imageArr = cv2.imread(os.path.join(path, image))
                updImage = resizeAndRescale(imageArr, 255)
                # updImage = cv2.resize(imageArr, (imgSize, imgSize))  # Resizing images to 200x200
                print("debug")
                trainingData.append([updImage, classNum])
            except Exception as e:
                pass


# createTrainingData()
# print(len(trainingData))
# random.shuffle(trainingData)  # Randomizes the training data in order to create unpredictability for the model

# x = []
# y = []
#
# for features, label in trainingData:
#     x.append(features)
#     y.append(label)
#
# x = np.array(x).reshape(-1, imgSize, imgSize, 3)
#
# # # Save training Data partition
# # pickleOut = open("x.pickle", "wb")
# # pickle.dump(x, pickleOut)
# # pickleOut.close()
# #
# # pickleOut = open("y.pickle", "wb")
# # pickle.dump(y, pickleOut)
# # pickleOut.close()
#
#
# pickleIn = open("x.pickle", "rb")
# x = pickle.load(pickleIn)
#
# pickleIn = open("y.pickle", "rb")
# y = pickle.load(pickleIn)





### DeepDream

# Download an image and read it into a NumPy array.
def download(url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)

# Normalize an image
def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

# Display an image
def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))

# image we will process
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

# Downsizing the image makes it easier to work with.
original_img = download(url, max_dim=500)
show(original_img)
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))

