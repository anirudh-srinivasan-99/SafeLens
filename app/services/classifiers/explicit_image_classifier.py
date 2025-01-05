"""This file is responsible for classifying if an image is fake or not."""
import os

import configparser
import cv2
import keras
from keras import backend as K
import numpy as np


class ExplicitImageClassifier:
    """
    This class is responsible for classifying if a given image is fake or not.
    Fake implies that the image was generated using generativeAI like deepfake etc.
    """
    def __init__(self, config_path: str):
        self._config = None
        self._models = []
        self._model_paths = [
            r'app\models\explicit_image_classfier\model_1.001-0.153.h5',
            r'app\models\explicit_image_classfier\model_2.003-0.165.h5',
            r'app\models\explicit_image_classfier\model_3.001-0.180.h5',
            r'app\models\explicit_image_classfier\model_4.002-0.201.h5'
        ]
        self.EXPLICIT_CLASSES = "Non-Porn Porn".split()

        # self._load_config(config_path)
        self._load_models()


    def _load_config(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config File not found at {config_path}.')
        self.config = 0


    def _load_models(self):
        for model_path in self._model_paths:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model path not found at {model_path}.')
            if os.path.splitext(model_path)[-1] in ('h5',):
                raise TypeError(f'Expects .h5 as file format, but got {model_path}.')
            self._models.append(keras.models.load_model(model_path))


    def _preprocess_input(self, img: np.ndarray, size: int = 229) -> np.ndarray:
        """
        Prepare an image for model input without saving it to a file.
        
        :param file_path: Path to the input image file.
        :param size: The target size of the image for the model (default is 224).
        :return: Preprocessed image ready for model prediction.
        """
        img = cv2.resize(img, (size, size))
        return img.reshape(-1, size, size, 3)


    def model_inference(self, img: np.ndarray) -> str:
        proc_img = self._preprocess_input(img)
        prediction = sum(model.predict(proc_img) for model in self._models)
        K.clear_session()
        pred_class = self.EXPLICIT_CLASSES[0 if np.argmax(prediction) in [0, 2, 4] else 1]
        print(pred_class)
        return pred_class


if __name__ == '__main__':
    img = cv2.imread(r'app\test_images\porn_189.jpg')
    explicit_img_classifier_obj = ExplicitImageClassifier('')
    explicit_img_classifier_obj.model_inference(img)