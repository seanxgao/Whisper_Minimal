"""
Unified Voice Activity Detection - Single VAD algorithm for all modes
"""

import numpy as np
from typing import List

class UnifiedVAD:
    """Unified Voice Activity Detection using adaptive min/max threshold"""
    
    def __init__(self, max_history: int = 50):
        """
        Initialize unified VAD
        
        Args:
            max_history: Maximum number of energy samples to keep for adaptive thresholding
        """
        self.max_history = max_history
        self.energy_history: List[float] = []
    
    def is_voice_activity(self, audio_data: np.ndarray) -> tuple[bool, str]:
        """
        Determine if audio contains voice activity using adaptive thresholding
        
        Args:
            audio_data: Audio data array
            
        Returns:
            Tuple of (is_voice, reason)
        """
        # Calculate RMS energy
        energy = np.mean(audio_data ** 2)
        
        # Update energy history for adaptive thresholding
        self.energy_history.append(energy)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
        
        # Need at least 10 samples to calculate adaptive threshold
        if len(self.energy_history) < 10:
            # Fallback to simple check during initial samples
            if energy < 1e-8:
                return False, "Complete silence (initializing)"
            return True, "Voice activity detected (initializing)"
        
        # Calculate adaptive threshold: (min + max) / 2
        min_energy = min(self.energy_history)
        max_energy = max(self.energy_history)
        
        # Avoid division by zero
        if max_energy <= min_energy:
            return True, "Voice activity detected (no variation)"
        
        # Calculate middle threshold
        middle_threshold = (min_energy + max_energy) / 2
        
        # Check if energy is above middle threshold
        if energy < middle_threshold:
            return False, f"Below adaptive threshold ({energy:.6f} < {middle_threshold:.6f})"
        
        # Check for extremely high noise (very high ZCR)
        zero_crossing_rate = np.mean(np.diff(np.sign(audio_data)) != 0)
        if zero_crossing_rate > 0.5:  # Very high threshold for noise
            return False, "High noise level detected"
        
        return True, f"Voice activity detected (above {middle_threshold:.6f})"
    
    def reset(self):
        """Reset VAD state"""
        self.energy_history.clear()

def run_vad(audio_data: np.ndarray, config: dict) -> tuple[bool, str]:
    """
    Simple VAD function for batch processing compatibility
    
    Args:
        audio_data: Audio data array
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_voice, reason)
    """
    # Get VAD settings from config
    vad_enabled = config.get("vad_enabled", True)
    vad_threshold = config.get("vad_threshold", 0.01)
    
    if not vad_enabled:
        return True, "VAD disabled"
    
    # Calculate RMS energy
    energy = np.mean(audio_data ** 2)
    
    # Check if energy is too low
    if energy < vad_threshold:
        return False, f"Energy too low ({energy:.6f} < {vad_threshold})"
    
    # Check for silence
    if energy < 1e-8:
        return False, "Complete silence"
    
    # Check for extremely high noise (very high ZCR)
    zero_crossing_rate = np.mean(np.diff(np.sign(audio_data)) != 0)
    if zero_crossing_rate > 0.5:  # Very high threshold for noise
        return False, "High noise level detected"
    
    return True, f"Voice activity detected (energy: {energy:.6f})"