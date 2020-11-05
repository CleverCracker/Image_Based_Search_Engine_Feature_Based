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
feature_arr = all_features()
feature_arr = np.array(feature_arr)
dists = np.linalg.norm(feature_arr-searchFe, axis=1)
ids = np.argsort(dists)[:10]

for id_img in ids:
    imgread = cv2.imread(finalData[id_img]['image_path'])
    plt.imshow(imgread)
    plt.show()
