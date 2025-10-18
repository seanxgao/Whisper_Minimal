"""
Audio recording using sounddevice
"""

import sounddevice as sd
import numpy as np
from typing import Optional, Callable

class Recorder:
    """Audio recorder using sounddevice"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize recorder
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_data = []
        self.stream = None
    
    def start_recording(self, callback: Optional[Callable] = None) -> bool:
        """
        Start audio recording
        
        Args:
            callback: Optional callback function for real-time processing
            
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_recording:
            return False
        
        try:
            self.audio_data = []
            self.is_recording = True
            
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Audio callback status: {status}")
                
                if self.is_recording:
                    self.audio_data.append(indata.copy())
                    
                    if callback:
                        callback(indata)
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                dtype=np.float32
            )
            
            self.stream.start()
            return True
            
        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """
        Stop audio recording and return audio data
        
        Returns:
            Audio data as numpy array or None if failed
        """
        if not self.is_recording:
            return None
        
        try:
            self.is_recording = False
            
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            if self.audio_data:
                audio_array = np.concatenate(self.audio_data, axis=0)
                return audio_array
            else:
                return None
                
        except Exception as e:
            print(f"Failed to stop recording: {e}")
            return None
    
    def get_audio_duration(self, audio_data: np.ndarray) -> float:
        """
        Calculate audio duration in seconds
        
        Args:
            audio_data: Audio data array
            
        Returns:
            Duration in seconds
        """
        if audio_data is None or len(audio_data) == 0:
            return 0.0
        
        return len(audio_data) / self.sample_rate
