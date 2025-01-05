"""
This file is responsible for classifying if an image is fake or not.
The classification determines whether the image is generated using generative AI, like Deepfake, etc.
"""

import os
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
    This class is responsible for classifying whether a given image is fake or not.
    Fake implies that the image was generated using generative AI like Deepfake, etc.

    Attributes:
        _config (str): Configuration settings (currently not used).
        _model (keras.Model): The pre-trained Keras model for classification.
        _model_path (str): Path to the pre-trained model file.
        FAKE_CLASSES (list): List of the two possible output classes ("Fake", "Non-Fake").
        _face_extractor_obj (FaceExtractor): An object used to extract faces from the image.
    """

    def __init__(self, config_path: str, face_extractor_obj: FaceExtractor):
        """
        Initializes the `FakeImageClassifier` class by loading the pre-trained model.

        :param config_path: Path to the configuration file (currently not used).
        :param face_extractor_obj: An instance of the `FaceExtractor` class to extract faces from images.
        """
        self._config = None
        self._model = None
        self._model_path = r'app\models\fake_image_classifier\Dataset_14k.01-0.22.h5'
        self.FAKE_CLASSES = ["Fake", "Non-Fake"]

        self._face_extractor_obj = face_extractor_obj
        # Load models upon initialization
        self._load_models()

    def _load_config(self, config_path: str) -> None:
        """
        Loads the configuration file (not currently used).

        :param config_path: Path to the configuration file.
        :raises FileNotFoundError: If the config file does not exist.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config File not found at {config_path}.')
        self.config = 0

    def _load_models(self) -> None:
        """
        Loads the pre-trained model from the specified file path.

        :raises FileNotFoundError: If the model file path does not exist.
        :raises TypeError: If the model file format is not `.h5`.
        """
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f'Model path not found at {self._model_path}.')
        if os.path.splitext(self._model_path)[-1] != '.h5':
            raise TypeError(f'Expects .h5 as file format, but got {self._model_path}.')
        with custom_object_scope({'FixedDropout': FixedDropout}):
            self._model = keras.models.load_model(self._model_path)

    def _preprocess_input(self, img: np.ndarray, size: int = 224) -> np.ndarray:
        """
        Preprocesses the input image by extracting faces, resizing, and preparing it for model prediction.

        :param img: Input image as a numpy array.
        :param size: Target size of the image for the model (default is 224).
        :return: Preprocessed image ready for model prediction.
        """
        # Extract the face from the input image
        face = self._face_extractor_obj.extract_faces(img)

        # Resize the face to the target size
        face = cv2.resize(face, (size, size))
        
        # Convert the face image to an array, normalize and expand its dimensions
        img = image.img_to_array(face)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img

    def model_inference(self, img: np.ndarray) -> str:
        """
        Performs model inference to classify the input image as either "Fake" or "Non-Fake".

        The method preprocesses the image, performs prediction using the loaded model, 
        and returns the predicted class.

        :param img: Input image as a numpy array.
        :return: The predicted class ("Fake" or "Non-Fake").
        """
        proc_img = self._preprocess_input(img)
        prediction = self._model.predict(proc_img)
        K.clear_session()  # Clear Keras session after inference to release memory
        # Determine the predicted class
        pred_class = self.FAKE_CLASSES[np.argmax(prediction)]
        print(pred_class)
        return pred_class


if __name__ == '__main__':
    # Example usage
    img = cv2.imread(r'app\test_images\fake_neutral.jpeg')
    face_extractor_obj = FaceExtractor('')
    fake_img_classifier_obj = FakeImageClassifier('', face_extractor_obj)
    fake_img_classifier_obj.model_inference(img)
