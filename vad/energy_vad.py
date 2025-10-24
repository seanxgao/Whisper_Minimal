"""
Simple Energy-based Voice Activity Detection (VAD)
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings

class EnergyVAD:
    """Energy-based voice activity detector"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_length_ms: float = 25.0,
                 hop_length_ms: float = 10.0,
                 alpha: float = 0.7,
                 min_energy_threshold: float = 1e-6):
        """
        Initialize energy VAD
        
        Args:
            sample_rate: Sample rate (Hz)
            frame_length_ms: Frame length (ms)
            hop_length_ms: Hop length (ms)
            alpha: Adaptive threshold sensitivity [0.3, 1.0]
            min_energy_threshold: Minimum energy threshold
        """
        self.sample_rate = sample_rate
        self.frame_length_ms = frame_length_ms
        self.hop_length_ms = hop_length_ms
        self.alpha = max(0.3, min(1.0, alpha))  # Limit alpha range
        self.min_energy_threshold = min_energy_threshold
        
        # Calculate frame parameters
        self.frame_len = int(0.001 * frame_length_ms * sample_rate)
        self.hop_len = int(0.001 * hop_length_ms * sample_rate)
        
        # Energy history
        self.energy_history: List[float] = []
        self.max_history_length = 100
        
        # State
        self.is_initialized = False
        self.current_threshold = 0.0
        
        # Initialization output removed for ultra-clean interface
    
    def _compute_frame_energy(self, frame: np.ndarray) -> float:
        """Compute frame energy"""
        if len(frame) == 0:
            return 0.0
        energy = np.mean(frame ** 2)
        return max(energy, self.min_energy_threshold)
    
    def _compute_adaptive_threshold(self) -> float:
        """Compute adaptive threshold"""
        if len(self.energy_history) < 2:
            return self.min_energy_threshold
        
        energy_array = np.array(self.energy_history)
        mean_energy = np.mean(energy_array)
        std_energy = np.std(energy_array)
        threshold = mean_energy + self.alpha * std_energy
        return max(threshold, self.min_energy_threshold)
    
    def _update_energy_history(self, energy: float):
        """Update energy history"""
        self.energy_history.append(energy)
        if len(self.energy_history) > self.max_history_length:
            self.energy_history.pop(0)
    
    def process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process audio and return VAD mask and energy sequence"""
        if len(audio_data) == 0:
            return np.array([]), np.array([])
        
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        num_frames = (len(audio_data) - self.frame_len) // self.hop_len + 1
        if num_frames <= 0:
            return np.array([]), np.array([])
        
        energy_sequence = np.zeros(num_frames)
        vad_mask = np.zeros(num_frames)
        
        for i in range(num_frames):
            start_idx = i * self.hop_len
            end_idx = start_idx + self.frame_len
            
            if end_idx > len(audio_data):
                break
            
            frame = audio_data[start_idx:end_idx]
            energy = self._compute_frame_energy(frame)
            energy_sequence[i] = energy
            self._update_energy_history(energy)
            
            threshold = self._compute_adaptive_threshold()
            self.current_threshold = threshold
            vad_mask[i] = 1.0 if energy > threshold else 0.0
        
        self.is_initialized = True
        return vad_mask, energy_sequence
    
    def is_voice_activity(self, audio_data: np.ndarray) -> Tuple[bool, str]:
        """Check if audio contains voice activity"""
        if len(audio_data) == 0:
            return False, "Empty audio data"
        
        vad_mask, _ = self.process_audio(audio_data)
        if len(vad_mask) == 0:
            return False, "No frames processed"
        
        voice_ratio = np.mean(vad_mask)
        is_voice = voice_ratio > 0.1
        
        if is_voice:
            reason = f"Voice detected (ratio: {voice_ratio:.2f})"
        else:
            reason = f"No voice detected (ratio: {voice_ratio:.2f})"
        
        return is_voice, reason
    
    
    def reset(self):
        """Reset VAD state"""
        self.energy_history.clear()
        self.current_threshold = 0.0
        self.is_initialized = False
