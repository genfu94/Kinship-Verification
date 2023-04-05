import cv2
import numpy as np
import numpy.typing as npt
from .face_detection import FacePart


# TODO: This should be a utility function
def compute_vector_angle_with_horizontal_axis(v):
    return np.degrees(np.arctan2(v[1], v[0])) - 180


class FaceAligner:
    def __init__(self, face_detector, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):

        self.face_detector = face_detector
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def _compute_desired_face_scale(self, eye_center_diff_vector):
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.linalg.norm(eye_center_diff_vector)
        desiredDist = (1.0 - 2 * self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        return scale
    
    def _compute_transformation_matrix(self, center, angle, scale):
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(center, angle, scale)
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - center[0])
        M[1, 2] += (tY - center[1])

        return M
    
    def _apply_transformation_matrix(self, image, M):
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    def align(self, image):
        self.face_detector.detect(image)
        left_eye_center = self.face_detector.get_face_part(FacePart.LEFT_EYE).mean(axis=0).astype("int")
        right_eye_center = self.face_detector.get_face_part(FacePart.RIGHT_EYE).mean(axis=0).astype("int")
        eye_center_diff_vector = right_eye_center - left_eye_center

        angle = compute_vector_angle_with_horizontal_axis(eye_center_diff_vector)
        scale = self._compute_desired_face_scale(eye_center_diff_vector)      
        eyes_center_median = (left_eye_center + right_eye_center)/2

        M = self._compute_transformation_matrix(eyes_center_median, angle, scale)

        return self._apply_transformation_matrix(image, M)