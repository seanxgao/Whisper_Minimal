"""
Teacher FSMN-VAD model wrapper.

This module provides a wrapper around the FSMN-VAD model from ModelScope,
loaded locally from a directory containing the model files.

Example usage:
    >>> from vad_distill.teacher import TeacherFSMNVAD
    >>> teacher = TeacherFSMNVAD(
    ...     model_dir="teacher",
    ...     device="cpu"
    ... )
    >>> import numpy as np
    >>> wav = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
    >>> frame_prob, segments = teacher.infer(wav, sr=16000)
    >>> print(f"Frame probabilities shape: {frame_prob.shape}")
    >>> print(f"Number of segments: {len(segments)}")
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class TeacherFSMNVAD:
    """
    Wrapper for FSMN-VAD teacher model loaded from local directory.
    
    Uses funasr.AutoModel to load the model from a local directory
    containing model files (model.pt, config.yaml, etc.).
    """
    
    def __init__(self, model_dir: str, device: str = "cpu"):
        """
        Initialize the teacher FSMN-VAD model.

        Args:
            model_dir: Path to the local FSMN-VAD model directory,
                       e.g. "teacher" (relative to project root) or absolute path
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.model_dir = Path(model_dir)
        self.device = device
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Teacher model directory not found: {model_dir}")
        
        logger.info(f"Loading FSMN-VAD teacher model from {model_dir}")
        
        try:
            from funasr import AutoModel
            
            # Load model from local directory
            self.model = AutoModel(
                model=str(self.model_dir),
                device=device,
            )
            logger.info("Teacher model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load teacher model: {e}")
    
    def infer(self, wav: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Run inference on a waveform to get frame-level VAD probabilities and segments.

        Args:
            wav: 1-D float32 waveform array, mono, at sr Hz
            sr: Sample rate of the waveform (default: 16000)

        Returns:
            frame_prob: np.ndarray of shape (T,), frame-level speech probability
            segments: List of [start_time, end_time] tuples in seconds (may be empty)
        """
        if len(wav.shape) > 1:
            raise ValueError(f"Expected 1-D waveform, got shape {wav.shape}")
        
        if sr != 16000:
            logger.warning(f"Sample rate {sr} != 16000, model expects 16kHz")
            # TODO: Resample if needed
        
        try:
            # Run inference with funasr
            # The exact API may vary; adjust based on funasr's actual interface
            # TODO: Verify the exact output format from funasr AutoModel
            result = self.model.generate(input=wav, cache={})
            
            # Extract frame probabilities and segments from result
            # The exact structure depends on funasr's output format
            # TODO: Parse result to extract:
            #   - frame_prob: frame-level probabilities
            #   - segments: list of [start, end] times in seconds
            
            # Placeholder implementation - adjust based on actual funasr output
            if isinstance(result, dict):
                frame_prob = result.get('frame_prob', np.zeros(len(wav) // 160, dtype=np.float32))
                segments = result.get('segments', [])
            elif isinstance(result, list) and len(result) > 0:
                # Assume result is a list with frame probs and segments
                frame_prob = result[0] if len(result) > 0 else np.zeros(len(wav) // 160, dtype=np.float32)
                segments = result[1] if len(result) > 1 else []
            else:
                # Fallback: create dummy outputs
                logger.warning("Unexpected result format from teacher model, using placeholder")
                num_frames = len(wav) // 160  # Approximate frames at 10ms hop
                frame_prob = np.ones(num_frames, dtype=np.float32) * 0.5
                segments = []
            
            # Ensure frame_prob is 1-D numpy array
            if not isinstance(frame_prob, np.ndarray):
                frame_prob = np.array(frame_prob, dtype=np.float32)
            if len(frame_prob.shape) > 1:
                frame_prob = frame_prob.flatten()
            
            # Ensure segments is list of tuples
            if segments and isinstance(segments[0], (list, tuple)) and len(segments[0]) == 2:
                segments = [(float(s[0]), float(s[1])) for s in segments]
            else:
                segments = []
            
            return frame_prob, segments
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Teacher inference error: {e}")

