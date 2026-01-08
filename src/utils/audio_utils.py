# src/utils/audio_utils.py

import soundfile as sf
import numpy as np

def save_audio(audio, file_path, sr=22050):
    """Save audio to WAV file."""
    try:
        sf.write(file_path, audio, sr)
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def load_audio(file_path, sr=22050):
    """Load audio from file."""
    try:
        audio, _ = sf.read(file_path)
        return audio
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def normalize_audio(audio, target_level=0.95):
    """Normalize audio to target level."""
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * target_level
    return audio
