from api.inference.emotion.classification import Infer
from pathlib import Path

if __name__ == "__main__":

    model = Infer()

    img_path = Path(__file__).parent / "example_imgs" / "happy_1.jpg"

    model.set_image(img_path)
    results = model.infer_image()
    print(f"Results for image: {img_path.name}")

    scores = Infer.get_numeric_scores(results)
