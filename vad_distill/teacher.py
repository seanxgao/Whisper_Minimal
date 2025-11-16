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
import os
import sys
from pathlib import Path
from typing import List, Tuple
import logging
from contextlib import redirect_stdout, redirect_stderr

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
            from tqdm import tqdm
            
            # Completely disable tqdm during model loading
            original_tqdm_disable = getattr(tqdm, 'disable', False)
            tqdm.disable = True
            original_tqdm_env = os.environ.get('TQDM_DISABLE')
            os.environ['TQDM_DISABLE'] = '1'
            
            try:
                # Load model from local directory
                # Disable internal progress bars by setting disable_tqdm if available
                with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull), redirect_stderr(fnull):
                        try:
                            self.model = AutoModel(
                                model=str(self.model_dir),
                                device=device,
                                disable_tqdm=True,  # Try to disable internal tqdm
                            )
                        except TypeError:
                            # If disable_tqdm is not supported, load without it
                            self.model = AutoModel(
                                model=str(self.model_dir),
                                device=device,
                            )
            finally:
                # Restore tqdm state
                tqdm.disable = original_tqdm_disable
                if original_tqdm_env is None:
                    os.environ.pop('TQDM_DISABLE', None)
                else:
                    os.environ['TQDM_DISABLE'] = original_tqdm_env
            
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
            # Completely disable tqdm globally before FunASR inference
            from tqdm import tqdm
            original_tqdm_disable = getattr(tqdm, 'disable', False)
            tqdm.disable = True
            
            # Set environment variable
            original_tqdm_env = os.environ.get('TQDM_DISABLE')
            os.environ['TQDM_DISABLE'] = '1'
            
            try:
                # Run inference with funasr
                # Suppress ALL output from FunASR
                with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull), redirect_stderr(fnull):
                        # FunASR VAD model typically returns a list of results
                        result = self.model.generate(input=wav, cache={})
            finally:
                # Restore tqdm state
                tqdm.disable = original_tqdm_disable
                if original_tqdm_env is None:
                    os.environ.pop('TQDM_DISABLE', None)
                else:
                    os.environ['TQDM_DISABLE'] = original_tqdm_env
            
            # FunASR VAD output format: usually a list, where each item is a dict
            # containing 'value' (segments) and possibly other metadata
            frame_prob = None
            segments = []
            
            if isinstance(result, list) and len(result) > 0:
                # Result is a list of dicts, each dict contains segment info
                for item in result:
                    if isinstance(item, dict):
                        # Extract segments from dict
                        # Common keys: 'value', 'timestamp', 'text', etc.
                        if 'value' in item:
                            # value might be a list of segments or a single segment
                            value = item['value']
                            if isinstance(value, list):
                                segments.extend(value)
                            elif isinstance(value, dict):
                                # Single segment as dict with 'start' and 'end' keys
                                if 'start' in value and 'end' in value:
                                    segments.append((value['start'], value['end']))
                        elif 'timestamp' in item:
                            # timestamp format: [[start, end], ...]
                            timestamp = item['timestamp']
                            if isinstance(timestamp, list):
                                segments.extend(timestamp)
                        # Try to extract from common segment formats
                        if 'start' in item and 'end' in item:
                            segments.append((item['start'], item['end']))
            
            elif isinstance(result, dict):
                # Result is a single dict
                if 'value' in result:
                    value = result['value']
                    if isinstance(value, list):
                        segments = value
                    elif isinstance(value, dict) and 'start' in value and 'end' in value:
                        segments = [(value['start'], value['end'])]
                elif 'timestamp' in result:
                    segments = result['timestamp']
                elif 'segments' in result:
                    segments = result['segments']
                frame_prob = result.get('frame_prob', None)
            
            # Normalize segments to list of (start, end) tuples
            normalized_segments = []
            for seg in segments:
                if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    # List/tuple format: [start, end] or (start, end)
                    normalized_segments.append((float(seg[0]), float(seg[1])))
                elif isinstance(seg, dict):
                    # Dict format: {'start': x, 'end': y} or {'begin_time': x, 'end_time': y}
                    start = seg.get('start') or seg.get('begin_time') or seg.get('begin')
                    end = seg.get('end') or seg.get('end_time') or seg.get('end')
                    if start is not None and end is not None:
                        normalized_segments.append((float(start), float(end)))
            
            segments = normalized_segments
            
            # Generate frame probabilities from segments if not provided
            # Calculate frame count to match librosa melspectrogram output
            # librosa uses: num_frames = (len(wav) - n_fft) // hop_length + 1
            # For 25ms frame length and 10ms hop at 16kHz:
            #   n_fft = 0.025 * 16000 = 400
            #   hop_length = 0.01 * 16000 = 160
            #   num_frames = (len(wav) - 400) // 160 + 1
            # This matches the frame count from log_mel() function
            from preprocessing.chunk_config import FRAME_LEN, FRAME_HOP
            n_fft = int(FRAME_LEN * sr)  # 400 at 16kHz
            hop_length = int(FRAME_HOP * sr)  # 160 at 16kHz
            num_frames = (len(wav) - n_fft) // hop_length + 1
            # Ensure non-negative
            num_frames = max(1, num_frames)
            if frame_prob is None:
                frame_prob = np.zeros(num_frames, dtype=np.float32)
                # Set probability to 1.0 in speech segments
                for start_time, end_time in segments:
                    start_frame = int(start_time * 100)  # 100 frames per second
                    end_frame = int(end_time * 100)
                    start_frame = max(0, min(start_frame, num_frames - 1))
                    end_frame = max(0, min(end_frame, num_frames))
                    frame_prob[start_frame:end_frame] = 1.0
            else:
                # Ensure frame_prob is 1-D numpy array
                if not isinstance(frame_prob, np.ndarray):
                    frame_prob = np.array(frame_prob, dtype=np.float32)
                if len(frame_prob.shape) > 1:
                    frame_prob = frame_prob.flatten()
                # Ensure correct length
                if len(frame_prob) != num_frames:
                    # Resample or pad/truncate
                    if len(frame_prob) < num_frames:
                        frame_prob = np.pad(frame_prob, (0, num_frames - len(frame_prob)), 'constant')
                    else:
                        frame_prob = frame_prob[:num_frames]
            
            return frame_prob, segments
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise RuntimeError(f"Teacher inference error: {e}")

