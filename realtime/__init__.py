"""
Realtime Pipeline Module for Portable version

This module provides realtime speech transcription capabilities including:
- Realtime voice activity detection
- Segment processing and transcription
- Text aggregation and final cleanup
- Pipeline coordination and management
"""

from .processor import SegmentProcessor, TextAggregator
from .pipeline import RealtimePipeline

__all__ = [
    'SegmentProcessor', 
    'TextAggregator',
    'RealtimePipeline'
]

__version__ = '1.0.0'
