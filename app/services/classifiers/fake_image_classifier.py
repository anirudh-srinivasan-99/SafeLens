"""This file is responsible for classifying if an image is fake or not."""
import os

import configparser
import cv2
import keras
from keras import backend as K
from keras.preprocessing import image
from keras.utils import custom_object_scope
import numpy as np

from app.services.extractor.face_extractor import FaceExtractor
from app.services.utils.utility import FixedDropout

class FakeImageClassifier:
    """
    This class is responsible for classifying if a given image is fake or not.
    Fake implies that the image was generated using generativeAI like deepfake etc.
    """
    def __init__(self, config_path: str, face_extractor_obj: FaceExtractor):
        self._config = None
        self._model = None
        self._model_path = r'app\models\fake_image_classifier\Dataset_14k.01-0.22.h5'
        self.FAKE_CLASSES = "Fake Non-Fake".split()

        self._face_extractor_obj = face_extractor_obj
        # self._load_config(config_path)
        self._load_models()


    def _load_config(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config File not found at {config_path}.')
        self.config = 0


    def _load_models(self):
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f'Model path not found at {self._model_path}.')
        if os.path.splitext(self._model_path)[-1] in ('h5',):
            raise TypeError(f'Expects .h5 as file format, but got {self._model_path}.')
        with custom_object_scope({'FixedDropout': FixedDropout}):
            self._model = keras.models.load_model(self._model_path)


    def _preprocess_input(self, img: np.ndarray, size: int = 224) -> np.ndarray:
        """
        Prepare an image for model input without saving it to a file.
        
        :param file_path: Path to the input image file.
        :param size: The target size of the image for the model (default is 224).
        :return: Preprocessed image ready for model prediction.
        """
        face = self._face_extractor_obj.extract_faces(img)
        
        face = cv2.resize(face, (size, size))
        img = image.img_to_array(face)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img


    def model_inference(self, img: np.ndarray) -> str:
        proc_img = self._preprocess_input(img)
        prediction = self._model.predict(proc_img)
        K.clear_session()
        pred = self.FAKE_CLASSES[np.argmax(prediction)]
        print(pred)
        return pred


if __name__ == '__main__':
    img = cv2.imread(r'app\test_images\fake_neutral.jpeg')
    face_extractor_obj = FaceExtractor('')
    fake_img_classifier_obj = FakeImageClassifier('', face_extractor_obj)
    fake_img_classifier_obj.model_inference(img)