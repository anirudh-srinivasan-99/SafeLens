"""
This file is responsible for extracting faces from input images using a pre-trained Caffe model.
It detects faces with confidence greater than 0.5 and returns the cropped face, or the original image if no face is detected.
"""

import os
import cv2
import numpy as np


class FaceExtractor:
    """
    This class is responsible for extracting faces from input images using a pre-trained face detection model.

    Attributes:
        model_arch_path (str): Path to the model architecture file (deploy.prototxt).
        model_weights_path (str): Path to the model weights file (weights.caffemodel).
        model (cv2.dnn.Net): The OpenCV DNN model used for face detection.
    """

    def __init__(self, config_path: str):
        """
        Initializes the `FaceExtractor` by loading the model architecture and weights.

        :param config_path: Path to the configuration file (currently not used).
        """
        self.model_arch_path = r'app\models\face_extraction\deploy.prototxt'
        self.model_weights_path = r'app\models\face_extraction\weights.caffemodel'
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Loads the face detection model from the specified architecture and weights files.

        :raises FileNotFoundError: If either the model architecture or weights file is not found.
        """
        if not os.path.exists(self.model_arch_path):
            raise FileNotFoundError(
                f'Model architecture for face extractor is not found at {self.model_arch_path}.'
            )
        if not os.path.exists(self.model_weights_path):
            raise FileNotFoundError(
                f'Model weights for face extractor is not found at {self.model_weights_path}.'
            )
        self.model = cv2.dnn.readNetFromCaffe(self.model_arch_path, self.model_weights_path)

    def extract_faces(self, img: np.ndarray) -> np.ndarray:
        """
        Extract faces from the input image. If a face is detected with confidence greater than 0.5,
        it will return the cropped face; otherwise, it will return the original image.

        :param img: The input image in which faces need to be detected.
        :type img: numpy.ndarray
        :return: The cropped face or the original image if no face is detected.
        :rtype: numpy.ndarray
        :raises ValueError: If the input image is invalid (None or empty).
        """
        if img is None or img.size == 0:
            raise ValueError("Invalid image input. Image cannot be None or empty.")

        (h, w) = img.shape[:2]
        blob = self._prepare_blob(img)

        self.model.setInput(blob)
        detections = self.model.forward()

        return self._process_detections(detections, img, w, h)

    def _prepare_blob(self, img: np.ndarray) -> np.ndarray:
        """
        Prepares the input image for the model by resizing and normalizing it.

        :param img: The input image to be processed.
        :type img: numpy.ndarray
        :return: The processed image blob ready for input to the model.
        :rtype: numpy.ndarray
        :raises ValueError: If the image is invalid or cannot be processed.
        """
        if img is None or img.size == 0:
            raise ValueError("Invalid image input. Image cannot be None or empty.")

        resized_img = cv2.resize(img, (300, 300))
        return cv2.dnn.blobFromImage(resized_img, 1.0, (300, 300), (104.0, 177.0, 123.0))

    def _process_detections(self, detections: np.ndarray, img: np.ndarray, w: int, h: int) -> np.ndarray:
        """
        Processes the model's detections and returns the detected face or the original image.

        :param detections: The output of the forward pass containing the detection results.
        :type detections: numpy.ndarray
        :param img: The original input image from which faces are being detected.
        :type img: numpy.ndarray
        :param w: The width of the original image.
        :type w: int
        :param h: The height of the original image.
        :type h: int
        :return: The cropped face or the original image if no face is detected.
        :rtype: numpy.ndarray
        """
        for i in range(detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype("int")

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                return img[start_y:end_y, start_x:end_x]

        return img


if __name__ == '__main__':
    # Example usage
    image = cv2.imread(r'app\test_images\porn_189.jpg')
    face_extractor_obj = FaceExtractor('')
    face = face_extractor_obj.extract_faces(image)
    cv2.imshow("face", face)
    cv2.waitKey(0)
