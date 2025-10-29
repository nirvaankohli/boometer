from api.inference.emotion.classification import Infer
from api.fear.scores.calculate import from_emotion_scores
from pathlib import Path
from PIL import Image

if __name__ == "__main__":

    model = Infer()

    img_path = Path(__file__).parent.parent / "example_imgs" / "happy_1.jpg"
    img = Image.open(img_path)

    model.set_image(img)
    results = model.predict()
    print(f"Results for image: {img_path.name}")

    scores = model.get_numeric_scores(results)
    print("Emotion Scores:", scores)

    fear_calculator = from_emotion_scores(emotion_scores=scores)
    fear_score = fear_calculator.calculate_fear_score()

    print(f"Calculated Fear Score: {fear_score:.4f}")
