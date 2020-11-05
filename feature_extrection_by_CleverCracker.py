from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from Data_Preprocessing import load_data
from feature_extrection import Features
import numpy as np
import cv2
import tensorflow as tf

# * File Directory Info
natural_images_dir = './images/'
labels = os.listdir(natural_images_dir)

# * Fetching Data of Image
x_data = []

for label in labels:
    path = natural_images_dir + label + '/'
    folder_images = os.listdir(path)
    for image_path in folder_images:
        image_load = cv2.imread(path+image_path)
       x_data.append(image_load)
x_data = np.array(x_data)
x_data.shape
x_data[1].shape

input_shape = (224, 224, 3)

# Model Initializtion

model = Sequential(
    [
        Conv2D(64, (3, 3), input_shape=input_shape,
               padding='same', activation='relu'),
        Conv2D(64, (3, 3),
               padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),

    ])
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam', metrics=['accuracy'])

img = x_data[6]
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
feature = model.predict(x)[0]
fea = feature/np.linalg.norm(feature)
print(fea.shape)

