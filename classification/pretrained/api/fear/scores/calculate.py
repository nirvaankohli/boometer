from typing import List
import itertools


class from_emotion_scores:

    def __init__(self, emotion_scores: dict = None):

        if emotion_scores is None:

            emotion_scores = {}

        self.emotion_weights = self.get_default_weights()
        self.emotion_scores = emotion_scores

    def set_emotion_scores(self, emotion_scores: dict):

        self.emotion_scores = emotion_scores

    def get_emotion_scores(self) -> dict:

        return self.emotion_scores

    def get_default_weights(self) -> dict:

        default_weights = {
            "anger": 2.5,
            "disgust": 5.0,
            "fear": 10.0,
            "happy": 2.0,
            "neutral": 2.5,
            "sad": 3.0,
            "surprise": 8.0,
        }

        return default_weights

    def get_possible_classes(self) -> List[str]:

        return [
            "anger",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ]

    def change_weights(self, new_weights: dict):

        self.emotion_weights = new_weights

    def calculate_fear_score(self) -> float:

        fear_score = 0.0

        top_items = list(
            itertools.islice(
                sorted(self.emotion_scores.items(), key=lambda x: x[1], reverse=True), 5
            )
        )
        emotion_scores = dict(top_items)
        for emotion, score in emotion_scores.items():

            weight = self.emotion_weights.get(emotion, 1.0)
            fear_score += (score / 10) * weight

        return fear_score
