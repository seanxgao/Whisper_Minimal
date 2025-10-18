"""
Configuration utilities
"""

import json
import os
from typing import Dict, Any

def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary with defaults
    """
    default_config = {
        "sample_rate": 16000,
        "channels": 1,
        "hotkey": "ctrl+space",
        "vad_enabled": True,
        "vad_backend": "simple",
        "vad_threshold": 0.01,
        "min_speech_ratio": 0.1,
        "ot_smoothing": False,
        "frame_length_ms": 30,
        "silence_duration_ms": 300,
        "vad_aggressiveness": 2,
        "debug_mode": False,
        "debug_audio_output": False,
        "debug_vad_details": False
    }
    
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}, using defaults")
        return default_config
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        
        # Merge with defaults
        config = {**default_config, **user_config}
        return config
        
    except Exception as e:
        print(f"Failed to load config: {e}, using defaults")
        return default_config

def load_api_key(key_file: str = None) -> str:
    """
    Load OpenAI API key from environment variable or file
    
    Args:
        key_file: Optional path to API key file (if not provided, uses environment variable)
        
    Returns:
        API key string
    """
    # First try environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # If no environment variable, try file
    if key_file and os.path.exists(key_file):
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
            
            if not api_key:
                raise ValueError("API key file is empty")
            
            return api_key
            
        except Exception as e:
            raise Exception(f"Failed to load API key from file: {e}")
    
    # If neither works, provide helpful error message
    raise Exception("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide a valid key file.")

def save_config(config: Dict[str, Any], config_file: str = "config.json") -> bool:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_file: Path to configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Failed to save config: {e}")
        return False
