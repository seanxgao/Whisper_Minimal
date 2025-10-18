"""
Keyboard input simulation using pynput
"""

import time
from pynput.keyboard import Controller
from typing import Optional

class KeyboardTyper:
    """Cross-platform keyboard input simulation"""
    
    def __init__(self):
        """Initialize keyboard controller"""
        self.keyboard = Controller()
    
    def type_text(self, text: str, delay: float = 0.01) -> bool:
        """
        Type text at current cursor position
        
        Args:
            text: Text to type
            delay: Delay between keystrokes (seconds)
            
        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            return False
        
        try:
            # Small delay to ensure focus
            time.sleep(0.1)
            
            # Type text with specified delay
            for char in text:
                self.keyboard.type(char)
                if delay > 0:
                    time.sleep(delay)
            
            return True
            
        except Exception as e:
            print(f"Keyboard typing failed: {e}")
            return False
    
    def type_text_fast(self, text: str) -> bool:
        """
        Type text quickly without delays
        
        Args:
            text: Text to type
            
        Returns:
            True if successful, False otherwise
        """
        return self.type_text(text, delay=0.0)
