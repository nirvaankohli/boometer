from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
import torch
import numpy as np


class Infer:

    def __init__(self, model="dima806/facial_emotions_image_detection", debug: bool = True):

        self.debug = debug
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
            self.print_debug(f"Error initializing model: {str(e)}")
            raise  # Re-raise the exception for proper error handling

    def print_debug(self, message: str):

        if self.debug:

            print(f"[DEBUG] {message}")

    def update_pipeline(self):

        try:

            self.pipe = pipeline(self.model_type, model=self.model)

        except Exception as e:

            self.print_debug(f"Error updating pipeline: {str(e)}")

    def change_model(self, new_model: str):

        self.model = new_model
        self.update_pipeline()

    def set_image(self, image):

        self.image = image

    def predict(self):
        self.print_debug("Running prediction...")

        try:
            if not hasattr(self, "image") or self.image is None:
                self.print_debug("No image set for prediction")
                return []

            self.print_debug(
                f"Image type: {type(self.image)}, size: {self.image.size if hasattr(self.image, 'size') else 'unknown'}"
            )
            results = self.pipe(self.image)
            self.print_debug(f"Raw prediction results: {results}")

            if not results:
                self.print_debug("Empty results from pipeline")
                return []

            return results

        except Exception as e:
            self.print_debug(f"Error during prediction: {str(e)}")
            import traceback

            self.print_debug(f"Traceback: {traceback.format_exc()}")
            return []

    def get_numeric_scores(self, results):
        try:
            if not results or not isinstance(results, list):
                self.print_debug(f"Invalid results format: {results}")
                return {}

            numeric_scores = {}
            for item in results:
                if isinstance(item, dict) and "label" in item and "score" in item:
                    try:
                        score = float(item["score"]) * 10
                        if not np.isnan(score) and score >= 0:
                            numeric_scores[item["label"].lower()] = score
                    except (ValueError, TypeError) as e:
                        self.print_debug(f"Error processing score for {item}: {e}")

            self.print_debug(f"Processed numeric scores: {numeric_scores}")
            return numeric_scores

        except Exception as e:
            self.print_debug(f"Error getting numeric scores: {str(e)}")
            import traceback

            self.print_debug(f"Traceback: {traceback.format_exc()}")
            return {}
