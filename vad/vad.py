"""
Voice Activity Detection Controller
"""

import numpy as np
from typing import Dict
from .energy_vad import EnergyVAD
from .threshold_vad import ThresholdVAD

def run_vad(audio_file: str, backend: str = "energy", sample_rate: int = 16000, 
            threshold: float = 0.01, aggressiveness: int = 3, alpha: float = 0.5, 
            mercy_time_ms: float = 300.0) -> dict:
    """Run VAD on audio file and return results with mercy time applied"""
    import wave
    import numpy as np
    
    # Read audio file
    with wave.open(audio_file, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Create VAD instance
    if backend == "energy":
        vad = EnergyVAD(sample_rate=sample_rate, alpha=alpha)
        vad_mask, energy_sequence = vad.process_audio(audio_data)
        
        # Apply mercy time
        vad_mask = apply_mercy_time(vad_mask, sample_rate, mercy_time_ms)
        
        speech_ratio = np.mean(vad_mask)
        silence_ratio = 1.0 - speech_ratio
        
        return {
            "speech_ratio": speech_ratio,
            "silence_ratio": silence_ratio,
            "vad_mask": vad_mask,
            "energy_sequence": energy_sequence,
            "is_voice": speech_ratio > 0.1
        }
    elif backend == "threshold":
        vad = ThresholdVAD(sample_rate=sample_rate, threshold=threshold)
        vad_mask, energy_sequence = vad.process_audio(audio_data)
        
        # Apply mercy time
        vad_mask = apply_mercy_time(vad_mask, sample_rate, mercy_time_ms)
        
        speech_ratio = np.mean(vad_mask)
        silence_ratio = 1.0 - speech_ratio
        
        return {
            "speech_ratio": speech_ratio,
            "silence_ratio": silence_ratio,
            "vad_mask": vad_mask,
            "energy_sequence": energy_sequence,
            "is_voice": speech_ratio > 0.1
        }
    else:
        # Legacy simple VAD
        energy = np.mean(audio_data ** 2)
        is_voice = energy > threshold
        speech_ratio = 1.0 if is_voice else 0.0
        silence_ratio = 1.0 - speech_ratio
        
        return {
            "speech_ratio": speech_ratio,
            "silence_ratio": silence_ratio,
            "is_voice": is_voice,
            "energy": energy
        }

def apply_mercy_time(vad_mask: np.ndarray, sample_rate: int, mercy_time_ms: float) -> np.ndarray:
    """Apply mercy time to VAD mask after VAD detection"""
    if mercy_time_ms <= 0 or len(vad_mask) == 0:
        return vad_mask
    
    # Calculate mercy frames (assuming 10ms hop length)
    hop_length_ms = 10.0
    mercy_frames = max(1, int(mercy_time_ms / hop_length_ms))
    
    # Find voice segments and extend them
    voice_indices = np.where(vad_mask == 1.0)[0]
    if len(voice_indices) > 0:
        # Extend each voice segment by mercy_frames
        for start_idx in voice_indices:
            end_idx = min(start_idx + mercy_frames, len(vad_mask))
            vad_mask[start_idx:end_idx] = 1.0
    
    return vad_mask
