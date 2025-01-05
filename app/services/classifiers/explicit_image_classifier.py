"""
This file is responsible for classifying if an image is explicit or not.
The classification determines whether the image is explicit content or non-explicit.
"""

import os
import cv2
import keras
from keras import backend as K
import numpy as np


class ExplicitImageClassifier:
    """
    This class is responsible for classifying if a given image is explicit or not.
    Explicit refers to images that are pornographic or contain explicit content.

    Attributes:
        _config (str): Configuration settings (currently not used).
        _models (list): List to store the loaded models.
        _model_paths (list): List of file paths for the models.
        EXPLICIT_CLASSES (list): List of the two possible output classes ("Non-Porn", "Porn").
    """

    def __init__(self, config_path: str):
        """
        Initializes the `ExplicitImageClassifier` class by loading the pre-trained models.

        :param config_path: Path to the configuration file (currently not used).
        """
        self._config = None
        self._models = []
        self._model_paths = [
            r'app\models\explicit_image_classfier\model_1.001-0.153.h5',
            r'app\models\explicit_image_classfier\model_2.003-0.165.h5',
            r'app\models\explicit_image_classfier\model_3.001-0.180.h5',
            r'app\models\explicit_image_classfier\model_4.002-0.201.h5'
        ]
        self.EXPLICIT_CLASSES = ["Non-Porn", "Porn"]

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
        Loads the pre-trained models from the specified file paths.

        :raises FileNotFoundError: If any model path does not exist.
        :raises TypeError: If the model file format is not `.h5`.
        """
        for model_path in self._model_paths:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model path not found at {model_path}.')
            if os.path.splitext(model_path)[-1] != '.h5':
                raise TypeError(f'Expects .h5 as file format, but got {model_path}.')
            self._models.append(keras.models.load_model(model_path))

    def _preprocess_input(self, img: np.ndarray, size: int = 229) -> np.ndarray:
        """
        Preprocesses the input image by resizing it and reshaping it for model prediction.

        :param img: Input image as a numpy array.
        :param size: Target size of the image for the model (default is 229).
        :return: Preprocessed image ready for model prediction.
        """
        img = cv2.resize(img, (size, size))
        return img.reshape(-1, size, size, 3)

    def model_inference(self, img: np.ndarray) -> str:
        """
        Performs model inference to classify the input image as either "Non-Porn" or "Porn".

        The method preprocesses the image, performs predictions using all the loaded models,
        aggregates the results, and maps them to one of the two classes.

        :param img: Input image as a numpy array.
        :return: The predicted class ("Non-Porn" or "Porn").
        """
        proc_img = self._preprocess_input(img)
        # Sum predictions from all models
        prediction = sum(model.predict(proc_img) for model in self._models)
        K.clear_session()  # Clear Keras session after inference to release memory
        # Determine the predicted class based on the aggregated prediction
        pred_class = self.EXPLICIT_CLASSES[0 if np.argmax(prediction) in [0, 2, 4] else 1]
        print(pred_class)
        return pred_class


if __name__ == '__main__':
    # Example usage
    img = cv2.imread(r'app\test_images\porn_189.jpg')
    explicit_img_classifier_obj = ExplicitImageClassifier(config_path="")
    explicit_img_classifier_obj.model_inference(img)
