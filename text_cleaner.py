"""
Text cleaning and reorganization using GPT-4o-mini
"""

import warnings
import openai
from typing import Optional

# Suppress Pydantic V1 compatibility warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater")

class TextCleaner:
    """GPT-based text cleaning and reorganization"""
    
    def __init__(self, api_key: str):
        """Initialize with OpenAI API key"""
        self.client = openai.OpenAI(api_key=api_key)
    
    def clean(self, text: str) -> Optional[str]:
        """
        Clean and reorganize text using GPT-4o-mini
        
        Args:
            text: Raw transcribed text
            
        Returns:
            Cleaned and reorganized text, or None if failed
        """
        if not text or not text.strip():
            return None
        
        try:
            print("Starting GPT text cleaning...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a professional text editor. Your task is to reorganize and improve the given text for better logical flow and clarity. 

Requirements:
1. Remove filler words, repetitions, and verbal tics (um, uh, like, you know, etc.)
2. Reorganize sentences for better logical flow and coherence
3. Correct technical terms and maintain accuracy
4. Preserve the original meaning and intent
5. Make the text more professional and readable
6. Handle mixed Chinese-English content appropriately
7. Keep the output concise but complete

Output only the cleaned text, no explanations or additional commentary."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            cleaned_text = response.choices[0].message.content.strip()
            print("GPT cleaning successful!")
            
            return cleaned_text if cleaned_text else text
            
        except Exception as e:
            print(f"GPT cleaning failed: {e}")
            return text  # Return original text if cleaning fails
