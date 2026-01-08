# src/models/__init__.py
"""
Deep learning models for emotional TTS
"""

from .tacotron2 import FinalEmotionalTacotron2, FinalHParams

__all__ = ['FinalEmotionalTacotron2', 'FinalHParams']
