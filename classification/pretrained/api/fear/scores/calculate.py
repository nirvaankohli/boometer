class from_emotion_scores:

    def __init__(self, emotion_scores: dict = None):

        if emotion_scores is None:

            emotion_scores = {}

        self.emotion_scores = emotion_scores

    def set_emotion_scores(self, emotion_scores: dict):

        self.emotion_scores = emotion_scores

    def get_emotion_scores(self) -> dict:

        return self.emotion_scores
