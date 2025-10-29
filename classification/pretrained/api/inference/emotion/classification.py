from transformers import pipeline


class Infer:

    def __init__(self, model="dima806/facial_emotions_image_detection"):

        self.model = model
        self.model_type = "image-classification"
        self.pipe = pipeline(self.model_type, model=self.model)

    def update_pipeline(self):

        self.pipe = pipeline(self.model_type, model=self.model)

    def change_model(self, new_model: str):

        self.model = new_model
        self.update_pipeline()

    def set_image(self, image):

        self.image = image

    def predict(self):

        results = self.pipe(self.image)

        return results

    def get_numeric_scores(self, results):

        numeric_scores = {}

        for i in results:

            numeric_scores[i["label"]] = i["score"] * 10

        return numeric_scores
