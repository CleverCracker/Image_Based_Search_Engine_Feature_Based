# * Import Libraries
from pathlib import Path
import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_natural_images_data():
    """
    Data
    """

    # * File Directory Info
    natural_images_dir = './natural_images/'
    labels = os.listdir(natural_images_dir)

    # * Fetching Data of Image
    x_data = []

    for label in labels:
        path = natural_images_dir + label + '/'
        folder_images = os.listdir(path)
        for image_path in folder_images:
            image = cv2.imread(path+image_path)
            x_data.append(image)
    x_data = np.array(x_data)
    return x_data


def feature_extrect(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    feature = model.predict(x)[0]
    return feature/np.linalg.norm(feature)


def features_Cal(imgData):
    """
    docstring
    """
    i = 0
    features = []
    for img in imgData:
        features.append(feature_extrect(img))
        print(i)
        i = i+1
    return features


def feature_save(imgDta):
    """
    Feature to Save
    """
    i = 0
    for img in imgDta:
        feature = feature_extrect(img)
        feature_path = Path('./features')/(str(i)+'.npy')
        np.save(feature_path, feature)
        i = i+1


def feature_load():
    """
    Load Feature
    """
    features = []
    for feature_path in Path('./features').glob('*.npy'):
        features.append(np.load(feature_path))
    return np.array(features)


def save_image_array(imgArr):
    """
    docstring
    """
    np.savetxt('image_array.csv', imgArr, delimiter=',')


def load_image_array():
    """
    docstring
    """
    return np.loadtxt('image_array.csv', delimiter=',')


def save_feature_array(feaArr):
    """
    docstring
    """
    np.savetxt('feature_array.csv', feaArr, delimiter=',')


def load_feature_array():
    """
    Feature Array Load
    """
    return np.loadtxt('feature_array.csv', delimiter=',')


base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
model = tf.keras.models.Model(
    inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

img_data = load_natural_images_data()
fea = features_Cal(img_data)
feature_array = np.asarray(fea)


feature_save(img_data)
# save_feature_array(feature_array)

# * Image Preprocessing

# searchImage = cv2.imread('searchDog.jpg')
searchFeature = feature_extrect(img_data[726])
# features = feature_load()
dists = np.linalg.norm(fea-searchFeature, axis=1)
ids = np.argsort(dists)[:10]
print('Upload Image')
plt.imshow(img_data[3])
plt.show()
print('Showing Similar')
for id_image in ids:
    plt.imshow(img_data[id_image])
    plt.show()


# for i in range(726):
#     plt.imshow(img_data[i])
#     plt.show()
#     i = i*3

# Feature Extrection Code


# OLD CODE
# fe1 = feature_extrect(x_data[1])
# fe2 = feature_extrect(x_data[2])
# dists = np.linalg.norm(fe1-fe2)
# ids = np.arange(dists)


# Testing Code Feature Extrect
# plt.imshow(image)
# plt.imshow(img)
# plt.imshow(x)
# x.shape
# plt.imshow(cv2.resize(x_data[1], (224, 224)))
# x = tf.keras.preprocessing.image.img_to_array(image)
# plt.imshow(image)
# print(np.expand_dims(x, axis=0))
