from feature_extrection import Features
from Data_Preprocessing import load_data
import cv2
import matplotlib.pyplot as plt
import numpy as np
data = load_data()


def all_features():
    """
    Return Array of Features
    """
    feature_arr = []
    for img_object in data['data']:
        feature_arr.append(img_object['feature'])
    return feature_arr


finalData = data['data']
searchImage = finalData[1800]
searchImg = cv2.imread(searchImage['image_path'])
plt.imshow(searchImg)
feature = Features()
searchFe = feature.feature_extrection(searchImg)
feature_arr = feature.all_features()
