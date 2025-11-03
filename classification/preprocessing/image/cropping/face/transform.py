import cv2
from typing import List, Tuple
import os
import numpy as np


class crop_face:

    def __init__(self, face_cascade_path: str, debug: bool = False):

        try:
            path_str = str(face_cascade_path)
        except Exception:
            path_str = face_cascade_path

        self.face_cascade = cv2.CascadeClassifier(path_str)
        self.debug = debug

        if self.face_cascade.empty():

            fallback = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            self.face_cascade = cv2.CascadeClassifier(fallback)
            if self.debug:
                print(
                    f"[DEBUG] Primary cascade load failed: {path_str}. Fallback to: {fallback}"
                )

        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade for face detection.")

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

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.equalizeHist(gray)


        tries = [
            (1.1, 4, (30, 30)),
            (1.05, 3, (24, 24)),
            (1.2, 5, (20, 20)),
        ]
        for sf, nbh, msz in tries:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=nbh, minSize=msz
            )
            if len(faces) > 0:
                if self.debug:
                    self.print_debug(
                        f"Faces detected with params scaleFactor={sf}, minNeighbors={nbh}, minSize={msz}: {len(faces)}"
                    )
                return faces

        if self.debug:
            self.print_debug("No faces detected in current frame")
        return []

    def crop_first_face(self, image: np.ndarray) -> np.ndarray:
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return None
        (x, y, w, h) = faces[0]
        cropped_face = image[y : y + h, x : x + w]
        return cropped_face
