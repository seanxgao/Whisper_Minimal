"""
VAD logging utilities
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

def setup_vad_logger(log_file: str = "logs/vad.log") -> logging.Logger:
    """
    Setup VAD logger with file and console output.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('vad')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_vad_result(logger: logging.Logger, 
                   vad_result: Dict[str, Any], 
                   audio_file: str = None) -> None:
    """
    Log VAD analysis results.
    
    Args:
        logger: Logger instance
        vad_result: VAD analysis result dictionary
        audio_file: Optional audio file path
    """
    
    speech_ratio = vad_result.get("speech_ratio", 0.0)
    silence_ratio = vad_result.get("silence_ratio", 0.0)
    segments = vad_result.get("segments", [])
    total_duration = vad_result.get("total_duration", 0.0)
    speech_duration = vad_result.get("speech_duration", 0.0)
    
    # Count speech segments
    speech_segments = sum(1 for seg in segments if seg.get("speech", False))
    silence_segments = len(segments) - speech_segments
    
    # Log main statistics
    logger.info(f"Speech: {speech_ratio*100:.1f}% | Silence: {silence_ratio*100:.1f}%")
    logger.info(f"Duration: {total_duration:.1f}s (speech: {speech_duration:.1f}s)")
    logger.info(f"Segments: {speech_segments} speech, {silence_segments} silence")
    
    # Log file info if provided
    if audio_file:
        filename = os.path.basename(audio_file)
        logger.info(f"File: {filename}")
    
    # Log detailed segments (only if few segments)
    if len(segments) <= 10:
        for i, seg in enumerate(segments):
            speech_type = "speech" if seg.get("speech", False) else "silence"
            logger.info(f"  Segment {i+1}: {seg['start']:.2f}s-{seg['end']:.2f}s ({speech_type})")

def log_vad_decision(logger: logging.Logger, 
                     speech_ratio: float, 
                     threshold: float, 
                     decision: str) -> None:
    """
    Log VAD decision (process or skip).
    
    Args:
        logger: Logger instance
        speech_ratio: Calculated speech ratio
        threshold: Minimum speech ratio threshold
        decision: Decision made ("process" or "skip")
    """
    
    logger.info(f"Decision: {decision.upper()} (speech: {speech_ratio*100:.1f}%, threshold: {threshold*100:.1f}%)")

def log_session_stats(logger: logging.Logger, 
                      total_recordings: int, 
                      processed_recordings: int, 
                      skipped_recordings: int) -> None:
    """
    Log session statistics.
    
    Args:
        logger: Logger instance
        total_recordings: Total number of recordings
        processed_recordings: Number of recordings processed
        skipped_recordings: Number of recordings skipped
    """
    
    if total_recordings > 0:
        skip_rate = skipped_recordings / total_recordings * 100
        logger.info("=== Session Statistics ===")
        logger.info(f"Total recordings: {total_recordings}")
        logger.info(f"Processed: {processed_recordings}")
        logger.info(f"Skipped: {skipped_recordings}")
        logger.info(f"Skip rate: {skip_rate:.1f}%")
        
        # Estimate cost savings (assuming $0.006/min for Whisper API)
        avg_duration = 30.0  # seconds, rough estimate
        saved_minutes = skipped_recordings * avg_duration / 60.0
        cost_saved = saved_minutes * 0.006
        logger.info(f"Estimated API cost saved: ${cost_saved:.3f}")
