"""
Main VAD segmenter interface
"""

import os
from typing import Dict, Any, List
from .backends import SimpleVADBackend, AdvancedSimpleVADBackend, OptimalTransportVADBackendWrapper
from .logger import setup_vad_logger, log_vad_result

def run_vad(file_path: str, 
            backend: str = "simple", 
            sample_rate: int = 16000, 
            threshold: float = 0.5,
            aggressiveness: int = 2,
            log_results: bool = True) -> Dict[str, Any]:
    """
    Run VAD analysis on audio file.
    
    Args:
        file_path: Path to audio file
        backend: VAD backend ("simple", "advanced", or "optimal_transport")
        sample_rate: Audio sample rate
        threshold: Speech detection threshold
        aggressiveness: Not used (kept for compatibility)
        log_results: Whether to log results
        
    Returns:
        Dictionary with VAD analysis results:
        {
            "segments": [{"start": 0.0, "end": 1.5, "speech": true}, ...],
            "speech_ratio": 0.83,
            "silence_ratio": 0.17,
            "total_duration": 2.04,
            "speech_duration": 1.69
        }
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Setup logger if requested
    logger = None
    if log_results:
        logger = setup_vad_logger()
    
    try:
        # Initialize VAD backend
        if backend == "simple":
            vad = SimpleVADBackend(sample_rate=sample_rate, energy_threshold=threshold)
        elif backend == "advanced":
            vad = AdvancedSimpleVADBackend(sample_rate=sample_rate, energy_threshold=threshold)
        elif backend == "webrtc":
            print("WebRTC VAD not available, falling back to simple VAD")
            vad = SimpleVADBackend(sample_rate=sample_rate, energy_threshold=threshold)
        elif backend == "silero":
            print("Silero VAD not available, falling back to simple VAD")
            vad = SimpleVADBackend(sample_rate=sample_rate, energy_threshold=threshold)
        elif backend == "optimal_transport" or backend == "ot":
            vad = OptimalTransportVADBackendWrapper(
                sample_rate=sample_rate, 
                energy_threshold=threshold,
                ot_threshold=0.3
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        # Run VAD detection
        segments = vad.detect(file_path)
        
        # Calculate statistics
        total_duration = 0.0
        speech_duration = 0.0
        
        for segment in segments:
            duration = segment["end"] - segment["start"]
            total_duration += duration
            
            if segment["speech"]:
                speech_duration += duration
        
        # Calculate ratios
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0
        silence_ratio = 1.0 - speech_ratio
        
        # Prepare result
        result = {
            "segments": segments,
            "speech_ratio": speech_ratio,
            "silence_ratio": silence_ratio,
            "total_duration": total_duration,
            "speech_duration": speech_duration,
            "backend": backend,
            "threshold": threshold
        }
        
        # Log results if requested
        if logger:
            log_vad_result(logger, result, file_path)
        
        return result
        
    except Exception as e:
        if logger:
            logger.error(f"VAD analysis failed: {e}")
        raise

def filter_speech_segments(segments: List[Dict[str, Any]], 
                          min_duration: float = 0.1,
                          margin: float = 0.3) -> List[Dict[str, Any]]:
    """
    Filter out very short speech segments and add margin to speech segments.
    
    Args:
        segments: List of VAD segments
        min_duration: Minimum duration for speech segments (seconds)
        margin: Margin to add before and after speech segments (seconds)
        
    Returns:
        Filtered segments list with margins
    """
    
    filtered = []
    for segment in segments:
        duration = segment["end"] - segment["start"]
        
        # Keep all silence segments and speech segments above minimum duration
        if not segment["speech"] or duration >= min_duration:
            if segment["speech"]:
                # Add margin to speech segments
                segment_with_margin = {
                    "start": max(0, segment["start"] - margin),
                    "end": segment["end"] + margin,
                    "speech": True
                }
                filtered.append(segment_with_margin)
            else:
                filtered.append(segment)
    
    return filtered

def merge_adjacent_segments(segments: List[Dict[str, Any]], 
                           max_gap: float = 0.1) -> List[Dict[str, Any]]:
    """
    Merge adjacent segments of the same type if gap is small.
    
    Args:
        segments: List of VAD segments
        max_gap: Maximum gap to merge (seconds)
        
    Returns:
        Merged segments list
    """
    
    if not segments:
        return []
    
    merged = [segments[0]]
    
    for current in segments[1:]:
        last = merged[-1]
        
        # Check if segments are same type and gap is small
        if (last["speech"] == current["speech"] and 
            current["start"] - last["end"] <= max_gap):
            # Merge segments
            merged[-1]["end"] = current["end"]
        else:
            # Add as separate segment
            merged.append(current)
    
    return merged

def get_speech_segments(segments: List[Dict[str, Any]], margin: float = 0.3) -> List[Dict[str, Any]]:
    """
    Extract only speech segments from VAD results with margin.
    
    Args:
        segments: List of VAD segments
        margin: Margin to add before and after speech segments (seconds)
        
    Returns:
        List of speech-only segments with margins
    """
    
    speech_segments = []
    for seg in segments:
        if seg["speech"]:
            segment_with_margin = {
                "start": max(0, seg["start"] - margin),
                "end": seg["end"] + margin,
                "speech": True
            }
            speech_segments.append(segment_with_margin)
    
    return speech_segments

def get_silence_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract only silence segments from VAD results.
    
    Args:
        segments: List of VAD segments
        
    Returns:
        List of silence-only segments
    """
    
    return [seg for seg in segments if not seg["speech"]]
