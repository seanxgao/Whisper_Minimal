"""
OpenAI Whisper API transcription
"""

import warnings
import openai  # type: ignore
import os
from typing import Optional

# Suppress Pydantic V1 compatibility warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater")

class Transcriber:
    """OpenAI Whisper API transcription service"""
    
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key"""
        self.client = openai.OpenAI(api_key=api_key)
    
    def transcribe(self, audio_file: str) -> Optional[str]:
        """
        Transcribe audio file using OpenAI Whisper API
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or None if failed
        """
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return None
        
        try:
            print("Transcribing with Whisper API...")
            
            with open(audio_file, "rb") as audio_file_obj:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file_obj,
                    response_format="text",
                    temperature=0.0
                )
            
            text = response.strip()
            return text if text else None
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            return None
