from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
import torch


class Infer:

    def __init__(self, model="Dc-4nderson/vit-emotion-classifier"):
        self.model = model
        self.model_type = "image-classification"

        try:
            # Load the model directly to CPU
            model = AutoModelForImageClassification.from_pretrained(
                self.model, torch_dtype=torch.float32, low_cpu_mem_usage=True
            ).to_empty(device="cpu")

            # Then load the processor
            processor = AutoImageProcessor.from_pretrained(self.model)
            self.pipe = pipeline(
                self.model_type,
                model=model,
                image_processor=processor,
                device="cpu",  # Explicitly use CPU
            )

        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise  # Re-raise the exception for proper error handling

    def update_pipeline(self):

        self.pipe = pipeline(self.model_type, model=self.model)

    def change_model(self, new_model: str):

        self.model = new_model
        self.update_pipeline()

    def set_image(self, image):

        self.image = image

    def predict(self):

        print("Running prediction...")
        try:
            results = self.pipe(self.image)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return []

        print("Prediction completed.")

        return results

    def get_numeric_scores(self, results):

        numeric_scores = {}

        for i in results:

            numeric_scores[i["label"]] = i["score"] * 10

        return numeric_scores
