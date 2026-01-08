# src/web_interface/app.py - Complete Enhanced Version

from flask import Flask, render_template, request, jsonify, send_file
import sys
import os
from pathlib import Path
import soundfile as sf
import numpy as np
from io import BytesIO
import base64
import tempfile
import subprocess
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.text_processing import clean_text, extract_text_from_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

project_root = Path(__file__).parent.parent.parent
app.config['UPLOAD_FOLDER'] = str(project_root / 'uploads')
app.config['AUDIO_FOLDER'] = str(project_root / 'audio_outputs')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
print(f"Audio folder: {app.config['AUDIO_FOLDER']}")


# ========== EMOTIONAL TTS ENGINE ==========

class EmotionalTTSEngine:
    """Multi-emotion TTS synthesis engine."""
    
    def __init__(self):
        print("Initializing Emotional TTS Engine...")
        self.sr = 22050
        self.use_espeak = self._check_espeak()
        print(f"Using: {'espeak' if self.use_espeak else 'pyttsx3'}")
    
    def _check_espeak(self):
        """Check if espeak is available."""
        try:
            subprocess.run(['espeak', '--version'], 
                         capture_output=True, check=True)
            return True
        except:
            return False
    
    def synthesize(self, text, emotion='neutral'):
        """Synthesize speech with emotional prosody."""
        print(f"Synthesizing with {emotion} emotion")
        
        if self.use_espeak:
            audio = self._synthesize_espeak(text, emotion)
        else:
            audio = self._synthesize_pyttsx3(text, emotion)
        
        return audio
    
    def _synthesize_espeak(self, text, emotion='neutral'):
        """Synthesize using espeak with emotion parameters."""
        try:
            import librosa
            
            print(f"  Synthesizing with espeak...")
            
            emotion_params = {
                'neutral': {'pitch': 50, 'speed': 150, 'gap': 0},
                'happy': {'pitch': 70, 'speed': 160, 'gap': 10},
                'angry': {'pitch': 30, 'speed': 180, 'gap': 0},
                'sad': {'pitch': 40, 'speed': 100, 'gap': 20},
                'surprise': {'pitch': 80, 'speed': 170, 'gap': 15}
            }
            
            params = emotion_params.get(emotion, emotion_params['neutral'])
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            cmd = [
                'espeak',
                '-w', temp_path,
                '-p', str(params['pitch']),
                '-s', str(params['speed']),
                '-g', str(params['gap']),
                text
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            audio, sr = sf.read(temp_path)
            
            if sr != 22050:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
            
            audio = self._apply_emotion_processing(audio, emotion)
            
            os.remove(temp_path)
            
            print(f"  Generated: {len(audio)/22050:.2f}s ({emotion})")
            
            return audio
        
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def _synthesize_pyttsx3(self, text, emotion='neutral'):
        """Synthesize using pyttsx3 with emotion modulation."""
        try:
            import pyttsx3
            import librosa
            
            print(f"  Synthesizing with pyttsx3...")
            
            engine = pyttsx3.init()
            
            emotion_settings = {
                'neutral': {'rate': 150, 'volume': 0.8},
                'happy': {'rate': 180, 'volume': 0.95},
                'angry': {'rate': 200, 'volume': 1.0},
                'sad': {'rate': 100, 'volume': 0.6},
                'surprise': {'rate': 190, 'volume': 0.95}
            }
            
            settings = emotion_settings.get(emotion, emotion_settings['neutral'])
            
            engine.setProperty('rate', settings['rate'])
            engine.setProperty('volume', settings['volume'])
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            audio, sr = sf.read(temp_path)
            
            if sr != 22050:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
            
            audio = self._apply_emotion_processing(audio, emotion)
            
            os.remove(temp_path)
            
            print(f"  Generated: {len(audio)/22050:.2f}s ({emotion})")
            
            return audio
        
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def _apply_emotion_processing(self, audio, emotion):
        """Apply DSP-based emotion processing to audio."""
        try:
            import scipy.signal
            
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            if emotion == 'happy':
                sos = scipy.signal.butter(4, [1000, 8000], 'bandpass', 
                                         fs=self.sr, output='sos')
                audio = audio + 0.3 * scipy.signal.sosfilt(sos, audio)
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
            
            elif emotion == 'angry':
                audio_distorted = np.tanh(audio * 2) * 0.5
                sos = scipy.signal.butter(4, [200, 2000], 'bandpass', 
                                         fs=self.sr, output='sos')
                boost = scipy.signal.sosfilt(sos, audio)
                audio = audio_distorted + 0.5 * boost
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
            
            elif emotion == 'sad':
                sos_low = scipy.signal.butter(4, 500, 'lowpass', 
                                              fs=self.sr, output='sos')
                audio = 0.7 * scipy.signal.sosfilt(sos_low, audio)
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
            
            elif emotion == 'surprise':
                lfo = 3 * np.sin(2 * np.pi * 5 * np.arange(len(audio)) / self.sr)
                audio = audio * (1 + 0.2 * lfo)
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95
            
            audio = self._post_process(audio)
            
            return audio
        
        except Exception as e:
            print(f"  Processing error: {e}")
            return audio
    
    def _post_process(self, audio):
        """Post-process audio."""
        audio = audio - np.mean(audio)
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.90
        
        fade_len = int(0.02 * self.sr)
        if len(audio) > 2 * fade_len:
            audio[:fade_len] *= np.linspace(0, 1, fade_len)
            audio[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        return audio


tts_engine = EmotionalTTSEngine()


# ========== FILE UPLOAD VALIDATION ==========

ALLOWED_EXTENSIONS = {'txt', 'docx', 'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file):
    """Validate uploaded file."""
    
    print(f"\nValidating file: {file.filename}")
    
    if not file.filename:
        return False, "No filename provided"
    
    if not allowed_file(file.filename):
        return False, f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size == 0:
        return False, "File is empty"
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Max size: 50MB"
    
    print(f"  File size: {file_size / 1024:.1f}KB")
    print(f"  File type: {file.filename.rsplit('.', 1)[1].upper()}")
    
    return True, "File valid"


# ========== FLASK ROUTES ==========

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        data = request.json
        text = data.get('text', '')
        emotion = data.get('emotion', 'neutral')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        text = clean_text(text)
        if len(text) > 1000:
            text = text[:1000]
        
        print(f"\nBackend: Processing request")
        print(f"Text: {text[:50]}...")
        print(f"Emotion: {emotion}")
        
        audio = tts_engine.synthesize(text, emotion)
        
        if audio is None:
            return jsonify({'error': 'Failed to synthesize audio'}), 500
        
        audio_filename = f'output_{emotion}.wav'
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        sf.write(audio_path, audio, 22050)
        
        print(f"Saved: {audio_filename} ({len(audio)/22050:.2f}s)")
        
        audio_buffer = BytesIO()
        sf.write(audio_buffer, audio, 22050, format='WAV')
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        
        waveform_data = audio[::10].tolist()
        
        return jsonify({
            'success': True,
            'audio_url': f'/audio/{audio_filename}',
            'audio_base64': f'data:audio/wav;base64,{audio_base64}',
            'text': text,
            'emotion': emotion,
            'duration': len(audio) / 22050,
            'waveform': waveform_data,
            'audio_filename': audio_filename
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("\n" + "="*60)
        print("FILE UPLOAD PROCESSING")
        print("="*60)
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file uploaded. Please select a file.'}), 400
        
        file = request.files['file']
        emotion = request.form.get('emotion', 'neutral')
        
        print(f"\nRequest details:")
        print(f"  - Emotion: {emotion}")
        print(f"  - File provided: {bool(file)}")
        
        if file.filename == '':
            print("No file selected")
            return jsonify({'error': 'No file selected. Please choose a file.'}), 400
        
        is_valid, message = validate_file(file)
        if not is_valid:
            print(f"Validation failed: {message}")
            return jsonify({'error': message}), 400
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        safe_filename = f"upload_{int(time.time())}.{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        print(f"\nSaving file...")
        print(f"  - Path: {file_path}")
        
        file.save(file_path)
        print(f"  File saved successfully")
        
        print(f"\nExtracting text...")
        try:
            text = extract_text_from_file(file_path)
        except Exception as e:
            print(f"Extraction error: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Failed to extract text: {str(e)}'}), 500
        
        if not text or len(text.strip()) < 10:
            print("No text extracted from file")
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'No readable text found in file'}), 400
        
        original_length = len(text)
        text = text[:10000]
        
        print(f"\nText extraction complete:")
        print(f"  - Characters extracted: {original_length}")
        print(f"  - Characters after limit: {len(text)}")
        print(f"  - First 100 chars: {text[:100]}...")
        
        try:
            os.remove(file_path)
            print(f"  Temporary file cleaned up")
        except:
            pass
        
        print("\nUpload processing successful!")
        print("="*60)
        
        return jsonify({
            'success': True,
            'text': text,
            'emotion': emotion,
            'filename': file.filename,
            'char_count': len(text)
        })
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/audio/<filename>')
def serve_audio(filename):
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
    
    if not os.path.exists(audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    
    return send_file(audio_path, mimetype='audio/wav')


if __name__ == '__main__':
    print("\n" + "="*70)
    print("EMOTIONAL TEXT-TO-SPEECH - MULTI-EMOTION SYNTHESIS")
    print("="*70)
    print(f"Project: {project_root}")
    print(f"TTS Engine: {'espeak' if tts_engine.use_espeak else 'pyttsx3'}")
    print(f"Emotions: Neutral, Happy, Angry, Sad, Surprise")
    print(f"Audio Output: {app.config['AUDIO_FOLDER']}")
    print("\nServer ready!")
    print("Open browser: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
