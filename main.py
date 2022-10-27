import cv2
import imutils
from src.face_recognition.face_alignment import FaceAligner
from src.face_recognition.face_detection import Dlib68LandmarksFaceDetector


image = cv2.imread('test_image.jpg')
image = imutils.resize(image, width=600)
fa = FaceAligner(Dlib68LandmarksFaceDetector(), desiredFaceWidth=256, desiredLeftEye=(0.3, 0.3))
aligned = fa.align(image)
cv2.imshow("ASD", aligned)
cv2.waitKey(0)