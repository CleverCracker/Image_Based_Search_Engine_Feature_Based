"""
    Data Load and Save String
"""

import numpy as np
import os
import json


def create_database():
    """
    Create Database And Loading Data
    """

    # * File Directory Info
    natural_images_dir = './images/'
    labels = os.listdir(natural_images_dir)

    # * Fetching Data of Image
    x_data = []
    data = {"data": []}

    i = 0
    for label in labels:
        path = natural_images_dir + label + '/'
        folder_images = os.listdir(path)
        for image_path in folder_images:
            imageObj = {
                'id': i,
                'image_path': path+image_path,
                'image_label': label,
                'feature': []
            }
            data['data'].append(imageObj)
            i = i+1

            # image = cv2.imread(path+image_path)
            # x_data.append(image)
    x_data = np.array(x_data)
    return data


def write_data(data, filename='./Database/data.json'):
    """
    Write Data to Json File
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)
    print('Writed...')
    return 1


def load_data(filename='./Database/data.json'):
    """
    Loading Json Database of Images Map
    """
    with open(filename) as json_file:
        return json.load(json_file)


def refresh_data():
    """
    ReWriting Data from Start
    """
    LoadData = create_database()
    write_data(LoadData)
