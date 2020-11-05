from Data_Preprocessing import load_data, write_data, refresh_data
import cv2
import numpy as np
import tensorflow as tf


class Features():
    """
    All Functionalities Using Feature Mapping and ETC
    """

    def __init__(self):
        """
        Model Seting
        """
        base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
        self.model = tf.keras.models.Model(
            inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        self.data = load_data()['data']

    def feature_extrection(self, img):
        """
        Return the Feature of Image
        """
        img = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
        preprocessing_img = tf.keras.preprocessing.image.img_to_array(img)
        preprocessing_img = np.expand_dims(preprocessing_img, axis=0)
        preprocessing_img = tf.keras.applications.vgg16.preprocess_input(
            preprocessing_img)
        feature_predict = self.model.predict(preprocessing_img)[0]
        feature_predict = feature_predict/np.linalg.norm(feature_predict)
        return feature_predict.tolist()

    def refresh_feature(self, img_object):
        """
        Check Feature are there are Not
        """
        if img_object['feature'] == []:
            img_object['feature'] = self.feature_extrection(
                cv2.imread(img_object['image_path']))
            print('Feature ' + str(img_object['id']) + " Refresh...")
            return img_object
        return -1

    def refresh_data_features(self):
        """
        Refresh All Data of Json Features
        """
        data = self.data
        for img_object in data:
            # if img_object['id'] == 4000:
            #     break
            refresh_object = self.refresh_feature(img_object)
            if refresh_object == -1:
                continue
            else:
                data[img_object['id']] = refresh_object
        json_data = {'data': data}
        write_data(json_data)
        print('Refreshed Completed')

    def all_features(self):
        """
        Return Array of Features
        """
        feature_arr = []
        for img_object in self.data:
            feature_arr.append(img_object['feature'])
        return feature_arr


if __name__ == "__main__":
    # refresh_data()

    feature = Features()
    feature.refresh_data_features()
