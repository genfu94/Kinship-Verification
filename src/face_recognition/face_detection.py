import numpy as np
import numpy.typing as npt
import cv2
from abc import ABC, abstractmethod
from enum import Enum
import dlib
import os


class FacePart(Enum):
    MOUTH = 1
    INNER_MOUTH = 2
    RIGHT_EYEBROW = 3
    LEFT_EYEBROW = 4
    RIGHT_EYE = 5
    LEFT_EYE = 6
    NOSE = 7
    JAW = 8


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray):
        pass

    @abstractmethod
    def get_face_part(self, part: FacePart) -> npt.NDArray[np.int_]:
        pass


class Dlib68LandmarksFaceDetector(FaceDetector):
    FACIAL_LANDMARKS_68_IDXS = {
        FacePart.MOUTH: slice(48, 68),
        FacePart.INNER_MOUTH: slice(60, 68),
        FacePart.RIGHT_EYEBROW: slice(17, 22),
        FacePart.LEFT_EYEBROW: slice(22, 27),
        FacePart.RIGHT_EYE: slice(36, 42),
        FacePart.LEFT_EYE: slice(42, 48),
        FacePart.NOSE: slice(27, 36),
        FacePart.JAW: slice(0, 17)
    }

    def __init__(self):
        dlib_68_landmark_predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_predictor_68_face_landmarks.dat')
        self.face_landmarks_predictor = dlib.shape_predictor(dlib_68_landmark_predictor_path)

    def _dlib_object_detection_to_ndarray(self, shape, dtype="int"):
        return np.array([[shape.part(i).x, shape.part(i).y] for i in range(0, shape.num_parts)], dtype=dtype)

    # TODO: Should this method return something?
    # TODO: Add some checks!!
    def detect(self, image: np.ndarray):
        self.original_image = image
        grayscaled_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        self.face_rect = detector(grayscaled_image, 2)[0]

        dlib_detection_result = self.face_landmarks_predictor(grayscaled_image, self.face_rect)
        self.face_landmarks = self._dlib_object_detection_to_ndarray(dlib_detection_result)


    def get_face_part(self, part: FacePart) -> npt.NDArray[np.int_]:
        if part not in Dlib68LandmarksFaceDetector.FACIAL_LANDMARKS_68_IDXS:
            return [] # TODO: should return None??
        
        return self.face_landmarks[Dlib68LandmarksFaceDetector.FACIAL_LANDMARKS_68_IDXS[part]]
