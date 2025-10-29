from transformers import pipeline
from pathlib import Path
from PIL import Image

if __name__ == "__main__":

    model = "dima806/facial_emotions_image_detection"
    model_type = "image-classification"

    pipe = pipeline(model_type, model)

    image_name = "happy_1.jpg"
    test_image_path = Path(__file__).parent.parent / "example_imgs" / image_name
    img = Image.open(test_image_path)

    results = pipe(img)

    print(f"Results for image: {image_name}")
    for r in results:
        print(f"Label: {r['label']}, Score: {r['score']:.4f}")
