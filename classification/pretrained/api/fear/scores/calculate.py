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
            "anger": 0.8,
            "disgust": 3.25,
            "fear": 5,
            "happy": 0.25,
            "neutral": 1.0,
            "sad": 2.5,
            "surprise": 4.0,
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

        for emotion, score in self.emotion_scores.items()[:5]:

            weight = self.emotion_weights.get(emotion, 1.0)

            fear_score += score * weight

        return fear_score
