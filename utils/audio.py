# ============================================================================
# utils/audio.py
# ============================================================================
"""
Audio transcription using Whisper
"""
import whisper
from typing import Optional

class VoiceSystem:
    """Handles audio transcription"""
    
    def __init__(self, model_size: str = "base"):
        print(f"🎤 Loading Whisper '{model_size}' model...")
        self.model = whisper.load_model(model_size)
        print("✅ Whisper ready")
    
    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file to text"""
        try:
            result = self.model.transcribe(audio_path, language='en')
            return result['text'].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return None