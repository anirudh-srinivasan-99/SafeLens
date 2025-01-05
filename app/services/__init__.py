from app.services.classifiers.explicit_image_classifier import ExplicitImageClassifier
from app.services.classifiers.fake_image_classifier import FakeImageClassifier
from app.services.extractor.face_extractor import FaceExtractor
from app.services.image_classifier import ImageClassifier

config_path = r'app\config\config.ini'

face_extractor_obj = FaceExtractor(config_path=config_path)
explicit_image_classifier_obj = ExplicitImageClassifier(config_path=config_path)
fake_image_classifier_obj = FakeImageClassifier(config_path=config_path, face_extractor_obj=face_extractor_obj)
image_classifier_obj = ImageClassifier(explicit_image_classifier_obj, fake_image_classifier_obj)

