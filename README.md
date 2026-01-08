# Major-Project
Created a TTS model based on Tacotron2

# Emotional Text-to-Speech with Deep Learning

Complete implementation of an Emotional Text-to-Speech system using Tacotron2.

## Features

- 5 Emotion Classes: Neutral, Happy, Angry, Sad, Surprise
- Tacotron2 deep learning model
- Griffin-Lim vocoder for audio generation
- Interactive Jupyter notebook
- Flask web interface
- File upload support (.txt, .docx, .pdf)
- Real-time audio visualization

## Quick Start

### 1. Setup
python setup.py

text

### 2. Run Jupyter Notebook
jupyter notebook notebooks/emotional_tts_complete.ipynb

text

### 3. Or Run Web Application
python run_web_app.py

text
Then open: http://localhost:5000

## Project Structure

emotional_tts_final/
├── setup.py # Setup script
├── requirements.txt # Dependencies
├── run_web_app.py # Web app launcher
├── config/ # Configuration files
├── src/ # Source code
│ ├── models/ # Tacotron2 model
│ ├── vocoder/ # Audio generation
│ ├── utils/ # Utilities
│ └── web_interface/ # Flask app
├── notebooks/ # Jupyter notebooks
├── checkpoints/ # Model checkpoints
└── audio_outputs/ # Generated audio

text

## Requirements

- Python 3.8+
- PyTorch 1.10+
- 8GB RAM minimum
- 5GB disk space

## Usage

See notebooks/emotional_tts_complete.ipynb for complete examples.

## Author

Information Science and Engineering Student
VTU 2022 Scheme

## License

MIT License
