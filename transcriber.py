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
    
    def transcribe(self, audio_file: str, with_timestamps: bool = False) -> Optional[str]:
        """
        Transcribe audio file using OpenAI Whisper API
        
        Args:
            audio_file: Path to audio file
            with_timestamps: If True, return text with timestamps
            
        Returns:
            Transcribed text or None if failed
        """
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return None
        
        try:
            print("Transcribing with Whisper API...")
            
            with open(audio_file, "rb") as audio_file_obj:
                if with_timestamps:
                    response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file_obj,
                        response_format="verbose_json",
                        temperature=0.0,
                        timestamp_granularities=["segment"]
                    )
                    # Format response with timestamps
                    return self._format_with_timestamps(response)
                else:
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
    
    def _format_with_timestamps(self, response) -> str:
        """
        Format verbose_json response with timestamps
        
        Args:
            response: Whisper API verbose_json response
            
        Returns:
            Formatted text with timestamps
        """
        if not hasattr(response, 'segments') or not response.segments:
            return response.text if hasattr(response, 'text') else ""
        
        formatted_lines = []
        for segment in response.segments:
            # Access attributes directly (TranscriptionSegment object)
            start_time = getattr(segment, 'start', 0)
            end_time = getattr(segment, 'end', 0)
            text = getattr(segment, 'text', '').strip()
            
            if text:
                # Format time as [HH:MM:SS]
                start_str = self._format_time(start_time)
                formatted_lines.append(f"[{start_str}] {text}")
        
        return "\n".join(formatted_lines)
    
    def _format_time(self, seconds: float) -> str:
        """
        Format seconds to [HH:MM:SS] format
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
