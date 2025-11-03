# Boometer

A real-time emotion analysis application that monitors fear reactions while watching YouTube videos. The application uses machine learning to detect facial emotions through your camera and provides live fear score tracking.

## Features

- **YouTube Integration**: Search and watch YouTube videos directly in the application
- **Real-time Emotion Detection**: Uses facial recognition and ML models to analyze emotions
- **Fear Score Calculation**: Calculates and tracks fear levels with weighted averages
- **Live Visualization**: Real-time charts showing fear score progression
- **Data Export**: CSV export functionality for analysis data

## Components

### Main Application
- **app.py**: Primary Streamlit application with video streaming and emotion analysis
- **youtube/app.py**: Simplified YouTube video player without emotion analysis

### Classification System
- **Pretrained Models**: Uses transformer-based emotion classification models
- **Image Processing**: Face detection and cropping using OpenCV and Haar cascades
- **Fear Calculation**: Custom algorithm to derive fear scores from emotion classifications

### Key Modules
- `classification/pretrained/api/inference/emotion/classification.py`: Core emotion inference engine
- `classification/pretrained/api/fear/scores/calculate.py`: Fear score calculation logic
- `classification/preprocessing/image/cropping/face/transform.py`: Face detection and preprocessing

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- Transformers (Hugging Face)
- PyTorch
- Plotly
- streamlit-webrtc
- youtubesearchpython

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install streamlit opencv-python transformers torch plotly streamlit-webrtc youtubesearchpython
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. **Search Videos**: Use the search interface to find YouTube videos
2. **Enable Camera**: Allow camera access for emotion detection
3. **Watch and Analyze**: The application will track your facial expressions and calculate fear scores in real-time
4. **View Results**: Monitor live charts showing your emotional response over time

## Technical Details

- Uses pre-trained emotion classification models from Hugging Face
- Implements weighted averaging for stable fear score calculation
- Processes video frames at configurable intervals to optimize performance
- Stores analysis data in CSV format for further processing

## Privacy

All emotion analysis is performed locally on your device. No video or emotion data is transmitted to external servers.
