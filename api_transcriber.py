#!/usr/bin/env python3
"""
API Voice Transcriber - Modular version
"""

import os
import time
import threading
import wave
import signal
import sys
import pyperclip
import warnings
from typing import Optional

# Suppress Pydantic V1 compatibility warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater")

# Import our modules
from config_utils import load_config, load_api_key
from recorder import Recorder
from transcriber import Transcriber
from text_cleaner import TextCleaner
from keyboard_typer import KeyboardTyper
from vad import run_vad

class App:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        # Load configuration
        self.config = load_config()
        self.api_key = load_api_key()
        
        # Initialize components
        self.recorder = Recorder(
            sample_rate=self.config.get("sample_rate", 16000),
            channels=self.config.get("channels", 1)
        )
        self.transcriber = Transcriber(self.api_key)
        self.text_cleaner = TextCleaner(self.api_key)
        self.keyboard_typer = KeyboardTyper()
        
        # VAD settings
        self.vad_enabled = self.config.get("vad_enabled", True)
        self.vad_backend = self.config.get("vad_backend", "simple")
        self.min_speech_ratio = self.config.get("min_speech_ratio", 0.1)
        self.vad_aggressiveness = self.config.get("vad_aggressiveness", 2)
        
        # Debug settings
        self.debug_mode = self.config.get("debug_mode", False)
        self.debug_audio_output = self.config.get("debug_audio_output", False)
        self.debug_vad_details = self.config.get("debug_vad_details", False)
        
        # Statistics tracking
        self.total_recordings = 0
        self.processed_recordings = 0
        self.skipped_recordings = 0
        
        # Control flags
        self.running = True
        self.recording = False
        
        # Setup signal handlers for clean exit
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("Voice Transcriber initialized!")
        print(f"VAD: {'Enabled' if self.vad_enabled else 'Disabled'} ({self.vad_backend})")
        print(f"Hotkey: {self.config.get('hotkey', 'ctrl+space')}")
        if self.debug_mode:
            print("Debug mode: ENABLED")
            print(f"  - Audio output: {'ON' if self.debug_audio_output else 'OFF'}")
            print(f"  - VAD details: {'ON' if self.debug_vad_details else 'OFF'}")
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.running = False
        self._cleanup()
        sys.exit(0)
    
    def start(self):
        """Start the main application loop"""
        hotkey = self.config.get("hotkey", "ctrl+space")
        
        try:
            import keyboard
            keyboard.add_hotkey(hotkey, self._toggle_recording)
            print(f"Press {hotkey} to start/stop recording. Press Ctrl+C to exit.")
            
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self._cleanup()
    
    def _toggle_recording(self):
        """Toggle recording state"""
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _start_recording(self):
        """Start recording"""
        if self.recording:
            return
        
        self.recording = True
        success = self.recorder.start_recording()
        
        if success:
            print("Recording started...")
        else:
            print("Failed to start recording")
            self.recording = False
    
    def _stop_recording(self):
        """Stop recording and process audio"""
        if not self.recording:
            return
        
        self.recording = False
        print("Stopping recording...")
        
        # Get audio data
        audio_data = self.recorder.stop_recording()
        
        if audio_data is None or len(audio_data) == 0:
            print("No audio data recorded")
            return
        
        # Check minimum duration
        duration = self.recorder.get_audio_duration(audio_data)
        if duration < 0.5:  # Less than 0.5 seconds
            print("Recording too short, skipping")
            return
        
        print(f"Audio duration: {duration:.1f}s")
        
        # Process audio in separate thread
        threading.Thread(target=self._process, args=(audio_data,), daemon=True).start()
    
    def _process(self, audio_data):
        """Process recorded audio"""
        # Save audio file
        os.makedirs("temp_files", exist_ok=True)
        temp_file = f"temp_files/audio_{int(time.time() * 1000)}.wav"
        
        # Debug: Save audio for analysis if enabled
        if self.debug_mode and self.debug_audio_output:
            debug_file = f"temp_files/debug_audio_{int(time.time() * 1000)}.wav"
            with wave.open(debug_file, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self.recorder.sample_rate)
                f.writeframes((audio_data * 32767).astype('int16').tobytes())
            print(f"Debug: Audio saved to {debug_file}")
        
        try:
            with wave.open(temp_file, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self.recorder.sample_rate)
                f.writeframes((audio_data * 32767).astype('int16').tobytes())
            
            self.total_recordings += 1
            
            # VAD analysis
            if self.vad_enabled:
                print("Running VAD analysis...")
                try:
                    vad_result = run_vad(
                        temp_file,
                        backend=self.vad_backend,
                        sample_rate=self.recorder.sample_rate,
                        threshold=self.config.get("vad_threshold", 0.01),
                        aggressiveness=self.vad_aggressiveness
                    )
                    
                    speech_ratio = vad_result["speech_ratio"]
                    silence_ratio = vad_result["silence_ratio"]
                    
                    print(f"Speech: {speech_ratio*100:.1f}% | Silence: {silence_ratio*100:.1f}%")
                    
                    # Debug: Show detailed VAD information
                    if self.debug_mode and self.debug_vad_details:
                        print(f"Debug VAD Details:")
                        print(f"  - Backend: {self.vad_backend}")
                        print(f"  - Threshold: {self.config.get('vad_threshold', 0.01)}")
                        print(f"  - Min speech ratio: {self.min_speech_ratio}")
                        print(f"  - Total segments: {len(vad_result['segments'])}")
                        if self.vad_backend in ["optimal_transport", "ot"]:
                            speech_segments = [s for s in vad_result['segments'] if s['speech']]
                            if speech_segments and 'debug_info' in speech_segments[0]:
                                print(f"  - OT Features: {speech_segments[0]['debug_info']}")
                    
                    if speech_ratio < self.min_speech_ratio:
                        print("Insufficient speech detected, skipping API calls")
                        self.skipped_recordings += 1
                        os.remove(temp_file)
                        return
                        
                except Exception as e:
                    print(f"VAD analysis failed: {e}")
                    print("Continuing with transcription...")
            
            self.processed_recordings += 1
            
            # Transcribe audio
            text = self.transcriber.transcribe(temp_file)
            
            if text:
                print("Sending text to GPT-4o-mini for cleaning...")
                cleaned = self.text_cleaner.clean(text)
                
                if cleaned:
                    print("GPT cleaning completed!")
                    print(f"Original: {text}")
                    print(f"Cleaned: {cleaned}")
                    
                    # Type text and copy to clipboard
                    self.keyboard_typer.type_text(cleaned)
                    pyperclip.copy(cleaned)
                    print("Text typed and copied to clipboard!")
                else:
                    print("GPT cleaning failed, using original text")
                    self.keyboard_typer.type_text(text)
                    pyperclip.copy(text)
            else:
                print("Transcription failed")
            
        except Exception as e:
            print(f"Processing failed: {e}")
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Could not remove temp file: {e}")
    
    def _cleanup(self):
        """Clean up resources and show statistics"""
        print("\nCleaning up...")
        
        # Stop any ongoing recording
        if self.recording:
            self.recorder.stop_recording()
        
        # Show session statistics
        if self.total_recordings > 0:
            print("\n=== Session Statistics ===")
            print(f"Total recordings: {self.total_recordings}")
            print(f"Processed: {self.processed_recordings}")
            print(f"Skipped: {self.skipped_recordings}")
            
            if self.total_recordings > 0:
                skip_rate = self.skipped_recordings / self.total_recordings * 100
                print(f"Skip rate: {skip_rate:.1f}%")
                
                # Estimate cost savings (rough calculation)
                estimated_cost_per_call = 0.006  # $0.006 per minute for Whisper
                estimated_savings = self.skipped_recordings * estimated_cost_per_call
                print(f"Estimated cost savings: ${estimated_savings:.3f}")
        
        # Clean up temp files
        try:
            import shutil
            if os.path.exists("temp_files"):
                shutil.rmtree("temp_files")
                print("Temporary files cleaned up")
        except Exception as e:
                print(f"Could not clean temp files: {e}")
        
        print("Goodbye!")

def main():
    """Main entry point"""
    try:
        app = App()
        app.start()
    except Exception as e:
        print(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()