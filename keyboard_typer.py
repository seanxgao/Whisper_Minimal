"""
Clipboard copy and paste helper
"""

import time
import pyperclip
import keyboard


class KeyboardTyper:
    """Copy text to clipboard and paste at cursor"""

    def __init__(self):
        """Initialize clipboard/paste utilities"""
        pass

    def type_text(self, text: str) -> bool:
        """
        Copy text to clipboard and paste with Ctrl+V

        Args:
            text: Text to paste

        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            return False

        try:
            pyperclip.copy(text)
            # brief delay to ensure clipboard is ready
            time.sleep(0.05)
            # send paste shortcut
            keyboard.send("ctrl+v")
            return True

        except Exception as e:
            print(f"Clipboard paste failed: {e}")
            return False
