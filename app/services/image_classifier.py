"""
This module defines the `ImageClassifier` class, which combines the functionality 
of two classifiers: `ExplicitImageClassifier` and `FakeImageClassifier`. The class
provides a method to perform inference on an image and returns a combined classification 
result, indicating whether the image is explicit or fake based on the outputs of the 
two individual classifiers.

It also includes an example of how to use the `ImageClassifier` by loading an image,
initializing the necessary objects, and performing the classification.

Classes:
    ImageClassifier: A class that integrates explicit and fake image classifiers 
                     for combined image classification results.
"""

import cv2
import numpy as np

from app.services.classifiers.explicit_image_classifier import ExplicitImageClassifier
from app.services.classifiers.fake_image_classifier import FakeImageClassifier
from app.services.extractor.face_extractor import FaceExtractor


class ImageClassifier:
    """
    This class is responsible for classifying an image based on two criteria:
    1. Whether the image is explicit or not (using `ExplicitImageClassifier`).
    2. Whether the image is fake or not (using `FakeImageClassifier`).
    
    It combines the predictions of these two classifiers into a single output.

    Attributes:
        explicit_image_classifier_obj (ExplicitImageClassifier): An object for classifying explicit content in images.
        fake_image_classifier_obj (FakeImageClassifier): An object for classifying fake content in images.
    """
    
    def __init__(
        self,
        explicit_image_classifier_obj: ExplicitImageClassifier,
        fake_image_classifier_obj: FakeImageClassifier
    ) -> None:
        """
        Initializes the `ImageClassifier` object with the provided classifiers.

        :param explicit_image_classifier_obj: An instance of the `ExplicitImageClassifier` for explicit content classification.
        :type explicit_image_classifier_obj: ExplicitImageClassifier
        :param fake_image_classifier_obj: An instance of the `FakeImageClassifier` for fake content classification.
        :type fake_image_classifier_obj: FakeImageClassifier
        """
        self.explicit_image_classifier_obj = explicit_image_classifier_obj
        self.fake_image_classifier_obj = fake_image_classifier_obj
    
    def model_inference(self, img: np.ndarray) -> str:
        """
        Perform inference on the provided image using both the explicit and fake image classifiers.

        This method returns the classification results for both explicit and fake content as a combined string.

        :param img: The input image on which inference is to be performed.
        :type img: numpy.ndarray
        :return: A string containing the combined results of explicit and fake content classification.
        :rtype: str
        """
        explicit_class = self.explicit_image_classifier_obj.model_inference(img)
        fake_class = self.fake_image_classifier_obj.model_inference(img)
        return f'{explicit_class} {fake_class}'


if __name__ == '__main__':
    # Example usage
    img = cv2.imread(r'app\test_images\porn_189.jpg')
    
    # Initialize the face extractor, explicit image classifier, and fake image classifier
    face_extractor_obj = FaceExtractor('')
    explicit_image_classifier_obj = ExplicitImageClassifier(config_path='')
    fake_image_classifier_obj = FakeImageClassifier(config_path='', face_extractor_obj=face_extractor_obj)
    
    # Instantiate the ImageClassifier with the classifiers
    image_classifier_obj = ImageClassifier(
        explicit_image_classifier_obj=explicit_image_classifier_obj,
        fake_image_classifier_obj=fake_image_classifier_obj
    )
    
    # Perform inference and print the results
    print(image_classifier_obj.model_inference(img))
