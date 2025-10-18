"""
Simple energy-based VAD implementation - no external dependencies
"""

import wave
import numpy as np
from typing import List, Dict, Any

class SimpleVAD:
    """Simple energy-based VAD - no external dependencies"""
    
    def __init__(self, sample_rate=16000, energy_threshold=0.01, frame_duration=0.1):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.frame_duration = frame_duration  # seconds
        self.frame_size = int(sample_rate * frame_duration)
        
    def detect(self, audio_file: str) -> List[Dict[str, Any]]:
        """Detect speech segments using energy threshold"""
        segments = []
        
        with wave.open(audio_file, 'rb') as wf:
            # Read audio data
            audio_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize to [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Process in frames
            current_time = 0.0
            current_speech = None
            speech_start = None
            
            for i in range(0, len(audio_array), self.frame_size):
                frame = audio_array[i:i + self.frame_size]
                
                # Calculate RMS energy
                if len(frame) > 0:
                    energy = np.sqrt(np.mean(frame ** 2))
                    is_speech = energy > self.energy_threshold
                else:
                    is_speech = False
                
                # Create segments
                if current_speech is None:
                    current_speech = is_speech
                    speech_start = current_time
                elif current_speech != is_speech:
                    # State change
                    segments.append({
                        "start": speech_start,
                        "end": current_time,
                        "speech": current_speech
                    })
                    current_speech = is_speech
                    speech_start = current_time
                
                current_time += self.frame_duration
            
            # Add final segment
            if speech_start is not None:
                segments.append({
                    "start": speech_start,
                    "end": current_time,
                    "speech": current_speech
                })
        
        return segments

class AdvancedSimpleVAD:
    """Enhanced simple VAD with better speech detection"""
    
    def __init__(self, sample_rate=16000, energy_threshold=0.01, 
                 zero_crossing_threshold=0.1, frame_duration=0.1):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.zero_crossing_threshold = zero_crossing_threshold
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        
    def _calculate_energy(self, frame):
        """Calculate RMS energy of frame"""
        if len(frame) == 0:
            return 0.0
        return np.sqrt(np.mean(frame ** 2))
    
    def _calculate_zero_crossing_rate(self, frame):
        """Calculate zero crossing rate"""
        if len(frame) < 2:
            return 0.0
        
        # Count sign changes
        sign_changes = np.sum(np.diff(np.sign(frame)) != 0)
        return sign_changes / (len(frame) - 1)
    
    def _is_speech(self, frame):
        """Determine if frame contains speech using energy and ZCR"""
        energy = self._calculate_energy(frame)
        zcr = self._calculate_zero_crossing_rate(frame)
        
        # Speech typically has higher energy and moderate ZCR
        energy_condition = energy > self.energy_threshold
        zcr_condition = 0.01 < zcr < 0.3  # Avoid very low (silence) and very high (noise) ZCR
        
        return energy_condition and zcr_condition
    
    def detect(self, audio_file: str) -> List[Dict[str, Any]]:
        """Detect speech segments using enhanced criteria"""
        segments = []
        
        with wave.open(audio_file, 'rb') as wf:
            # Read audio data
            audio_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize to [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Process in frames
            current_time = 0.0
            current_speech = None
            speech_start = None
            
            for i in range(0, len(audio_array), self.frame_size):
                frame = audio_array[i:i + self.frame_size]
                is_speech = self._is_speech(frame)
                
                # Create segments
                if current_speech is None:
                    current_speech = is_speech
                    speech_start = current_time
                elif current_speech != is_speech:
                    # State change
                    segments.append({
                        "start": speech_start,
                        "end": current_time,
                        "speech": current_speech
                    })
                    current_speech = is_speech
                    speech_start = current_time
                
                current_time += self.frame_duration
            
            # Add final segment
            if speech_start is not None:
                segments.append({
                    "start": speech_start,
                    "end": current_time,
                    "speech": current_speech
                })
        
        return segments
