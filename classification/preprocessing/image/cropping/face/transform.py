import cv2
from typing import List
import numpy as np


class crop_face:

    def __init__(self, face_cascade_path: str, debug: bool = False):

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.debug = debug

    def print_debug(self, message: str):

        if self.debug:

            print(f"[DEBUG] {message}")

    def crop(self, image: np.ndarray) -> List[np.ndarray]:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20)
        )

        cropped_faces = []

        for x, y, w, h in faces:

            cropped_face = image[y : y + h, x : x + w]

            cropped_faces.append(cropped_face)

        return cropped_faces

    def detect_faces(self, image: np.ndarray) -> List[tuple]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20)
        )
        return faces

    def crop_first_face(self, image: np.ndarray) -> np.ndarray:
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return None
        (x, y, w, h) = faces[0]
        cropped_face = image[y : y + h, x : x + w]
        return cropped_face
