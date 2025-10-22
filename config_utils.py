"""
Simple configuration utilities - core functionality only
"""

import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """
    Load default configuration - core settings only
    
    Returns:
        Configuration dictionary with essential defaults
    """
    return {
        "sample_rate": 16000,
        "channels": 1,
        "hotkey": "ctrl+space",
        "vad_enabled": True,
        "vad_threshold": 0.01
    }

def load_api_key() -> str:
    """
    Load OpenAI API key from environment variable
    
    Returns:
        API key string
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    raise Exception("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

