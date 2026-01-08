# src/vocoder/vocoder.py - Complete Working Vocoder

import numpy as np
import librosa

class SimplifiedVocoder:
    """Simplified vocoder using librosa's Griffin-Lim algorithm."""
    
    def __init__(self, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Create mel filter bank
        self.mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        
        print(f"✅ Vocoder initialized:")
        print(f"  - Sample rate: {sr} Hz")
        print(f"  - FFT size: {n_fft}")
        print(f"  - Hop length: {hop_length}")
        print(f"  - Mel channels: {n_mels}")
    
    def decode(self, mel_spectrogram, griffin_lim_iter=100):
        """
        Convert mel-spectrogram to audio waveform.
        
        Args:
            mel_spectrogram: Mel-spectrogram array [n_mels, time] or [batch, time, n_mels]
            griffin_lim_iter: Number of Griffin-Lim iterations
        
        Returns:
            wav: Audio waveform
        """
        print("  🎵 Converting mel-spectrogram to audio...")
        
        try:
            # Handle batch dimension
            if mel_spectrogram.ndim == 3:
                mel_spectrogram = mel_spectrogram[0]
                print(f"    Removed batch dimension")
            
            mel_spec_shape = mel_spectrogram.shape
            print(f"    Input mel shape: {mel_spec_shape}")
            
            # Transpose if needed (should be [n_mels, time])
            if mel_spec_shape[0] > mel_spec_shape[1]:
                mel_spectrogram = mel_spectrogram.T
                print(f"    Transposed to: {mel_spectrogram.shape}")
            
            # Ensure correct shape
            if mel_spectrogram.shape[0] != self.n_mels:
                print(f"   ⚠️ Warning: Expected {self.n_mels} mel bins, got {mel_spectrogram.shape[0]}")
            
            # Convert db to power
            print(f"    Converting db to power...")
            mel_power = librosa.db_to_power(mel_spectrogram)
            
            # Convert mel to linear spectrogram
            print(f"    Converting mel to linear spectrogram...")
            D = librosa.feature.inverse.mel_to_stft(
                mel_power,
                sr=self.sr,
                n_fft=self.n_fft
            )
            
            print(f"    Linear spectrogram shape: {D.shape}")
            
            # Griffin-Lim reconstruction
            print(f"    Running Griffin-Lim algorithm ({griffin_lim_iter} iterations)...")
            wav = librosa.griffinlim(
                np.abs(D),
                n_iter=griffin_lim_iter,
                hop_length=self.hop_length,
                momentum=0.99
            )
            
            print(f"    Generated {len(wav)} samples ({len(wav)/self.sr:.2f}s)")
            
            # Post-processing
            print(f"    Post-processing audio...")
            wav = self._post_process(wav)
            
            print(f"    ✅ Final audio: {len(wav)/self.sr:.2f}s")
            
            return wav
        
        except Exception as e:
            print(f"    ❌ Error in decode: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _post_process(self, wav):
        """Post-process audio for better quality."""
        # Remove DC offset
        wav = wav - np.mean(wav)
        
        # Normalize
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val * 0.95
        
        # Apply fade in/out
        fade_len = int(0.01 * self.sr)  # 10ms fade
        if len(wav) > 2 * fade_len:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            wav[:fade_len] *= fade_in
            wav[-fade_len:] *= fade_out
        
        return wav
