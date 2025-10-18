"""
VAD backend implementations
"""

import wave
import numpy as np
from typing import List, Dict, Any
from .simple_vad import SimpleVAD, AdvancedSimpleVAD
from .ot_vad import OptimalTransportVADBackend

# Removed WebRTC and Silero VAD implementations due to dependency issues
# WebRTC VAD requires Visual C++ compiler on Windows
# Silero VAD requires torch, torchaudio, and silero-vad packages

class SimpleVADBackend:
    """Simple energy-based VAD backend - no external dependencies"""
    
    def __init__(self, sample_rate=16000, energy_threshold=0.01):
        self.vad = SimpleVAD(sample_rate=sample_rate, energy_threshold=energy_threshold)
    
    def detect(self, audio_file: str) -> List[Dict[str, Any]]:
        """Detect speech segments using simple energy-based method"""
        return self.vad.detect(audio_file)

class AdvancedSimpleVADBackend:
    """Enhanced simple VAD backend with energy + zero crossing rate"""
    
    def __init__(self, sample_rate=16000, energy_threshold=0.01, 
                 zero_crossing_threshold=0.1):
        self.vad = AdvancedSimpleVAD(
            sample_rate=sample_rate, 
            energy_threshold=energy_threshold,
            zero_crossing_threshold=zero_crossing_threshold
        )
    
    def detect(self, audio_file: str) -> List[Dict[str, Any]]:
        """Detect speech segments using enhanced criteria"""
        return self.vad.detect(audio_file)

class OptimalTransportVADBackendWrapper:
    """OT-VAD backend wrapper"""
    
    def __init__(self, sample_rate=16000, energy_threshold=0.01, 
                 ot_threshold=0.3, frame_duration=0.1):
        self.vad = OptimalTransportVADBackend(
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
            ot_threshold=ot_threshold,
            frame_duration=frame_duration
        )
    
    def detect(self, audio_file: str) -> List[Dict[str, Any]]:
        """Detect speech segments using Optimal Transport"""
        return self.vad.detect(audio_file)
