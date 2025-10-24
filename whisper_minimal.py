#!/usr/bin/env python3
"""
API Voice Transcriber
"""

import os
import time
import threading
import wave
import signal
import sys
import pyperclip
import warnings
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore")

import os
from recorder import Recorder
from transcriber import Transcriber
from text_cleaner import TextCleaner
from keyboard_typer import KeyboardTyper
from vad.vad import run_vad
from realtime import RealtimePipeline

class App:
    """Main application class"""
    
    def __init__(self, preset: Optional[str] = None, config_file: Optional[str] = None):
        """Initialize the application"""
        # Default configuration
        self.config = {
            "sample_rate": 16000,
            "channels": 1,
            "hotkey": "ctrl+space",
            "vad_enabled": True,
            "vad_backend": "energy",
            "vad_mercy_time_ms": 300,  # Mercy time for all VAD algorithms
            "realtime_mode": True,
            "debug_mode": False
        }
        
        # Load configuration from JSON file if provided
        if config_file and os.path.exists(config_file):
            import json
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
                    if self.config.get('debug_mode', False):
                        print(f"Loaded configuration from: {config_file}")
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")
                print("Using default configuration")
        
        # Load API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("OpenAI API key not found in environment variables.")
            self.api_key = input("Please enter your OpenAI API key: ").strip()
            if not self.api_key:
                raise Exception("No API key provided. Exiting.")
            # Set temporarily for this session only
            os.environ["OPENAI_API_KEY"] = self.api_key
        
        self.last_hotkey_time = 0
        self.hotkey_debounce_ms = 500  # 500ms debounce time
    
        
        # Initialize components
        self.recorder = Recorder(
            sample_rate=self.config.get("sample_rate", 16000),
            channels=self.config.get("channels", 1)
        )
        self.transcriber = Transcriber(self.api_key)
        self.text_cleaner = TextCleaner(self.api_key)
        self.keyboard_typer = KeyboardTyper()
        
        # Initialize realtime pipeline if enabled
        self.realtime_mode = self.config.get("realtime_mode", True)
        if self.realtime_mode:
            self.realtime_pipeline = RealtimePipeline(self.config, self.api_key)
        else:
            self.realtime_pipeline = None
        
        # VAD settings
        self.vad_enabled = self.config.get("vad_enabled", True)
        self.vad_backend = self.config.get("vad_backend", "energy")
        self.vad_threshold = self.config.get("vad_threshold", 0.01)
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
        
        # Ultra-clean initialization - only show hotkey
        print(f"Hotkey: {self.config.get('hotkey', 'ctrl+space')}")
        if self.debug_mode:
            print("Debug mode: ENABLED")
            print(f"VAD: {self.vad_backend}")
            print(f"Mode: {'Realtime Pipeline' if self.realtime_mode else 'Traditional Batch'}")
            print(f"VAD: {'Enabled' if self.vad_enabled else 'Disabled'} ({self.vad_backend})")
            print(f"  - Audio output: {'ON' if self.debug_audio_output else 'OFF'}")
            print(f"  - VAD details: {'ON' if self.debug_vad_details else 'OFF'}")
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        if False:  # Disable signal received message
            print(f"\nReceived signal {signum}, shutting down...")
        self.running = False
        # Don't call _cleanup here, let finally block handle it
        sys.exit(0)
    
    def start(self):
        """Start the main application loop"""
        try:
            import keyboard
            # Register three hotkeys (order matters - more specific first)
            keyboard.add_hotkey("ctrl+shift+space", self._start_session)
            keyboard.add_hotkey("shift+ctrl+space", self._end_session)
            keyboard.add_hotkey("ctrl+space", self._cut_segment)
            
            print("=== Segment Recording Mode ===")
            print("Ctrl+Shift+Space: Start/End recording session")
            print("Ctrl+Space: Cut current segment, start next")
            print("Shift+Ctrl+Space: End session, output final text")
            print("Ctrl+C: Exit program")
            print()
            
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            if False:  # Disable shutting down message
                print("\nShutting down...")
        finally:
            self._cleanup()
    
    def _debounce_hotkey(self):
        """Check if hotkey should be processed (debouncing)"""
        current_time = time.time() * 1000  # Convert to milliseconds
        if current_time - self.last_hotkey_time < self.hotkey_debounce_ms:
            return False
        self.last_hotkey_time = current_time
        return True
    
    def _start_session(self):
        """Start recording session or end if already recording"""
        if not self._debounce_hotkey():
            return
            
        if self.recording:
            # If already recording, treat this as end session
            if False:  # Disable ending recording session message
                print("Ending recording session...")
            self._stop_recording()
            return
        
        if False:  # Disable starting recording session message
            print("Starting recording session...")
        self._start_recording()
    
    def _cut_segment(self):
        """Cut current segment and start next"""
        if not self._debounce_hotkey():
            return
            
        if not self.recording:
            if False:  # Disable no active recording session message
                print("No active recording session")
            return
        
        if False:  # Disable cutting segment message
            print("Cutting current segment...")
        if self.realtime_mode and self.realtime_pipeline:
            # Cut current segment and process
            self.realtime_pipeline.cut_and_process_segment()
        else:
            # Traditional mode: stop recording
            self._stop_recording()
    
    def _end_session(self):
        """End recording session"""
        if not self._debounce_hotkey():
            return
            
        if not self.recording:
            if False:  # Disable no active recording session message
                print("No active recording session")
            return
        
        if False:  # Disable ending recording session message
            print("Ending recording session...")
        self._stop_recording()
    
    def _start_recording(self):
        """Start recording"""
        if self.recording:
            return
        
        if self.realtime_mode and self.realtime_pipeline:
            # Use realtime pipeline mode
            success = self.realtime_pipeline.start_recording()
            if success:
                self.recording = True
                if False:  # Disable realtime recording started message
                    print("Realtime recording started...")
            else:
                print("Failed to start realtime recording")
        else:
            # Use traditional batch mode
            self._start_traditional_recording()
    
    def _stop_recording(self):
        """Stop recording and process audio"""
        if not self.recording:
            return
        
        self.recording = False
        
        if self.realtime_mode and self.realtime_pipeline:
            # Use realtime pipeline mode
            if False:  # Disable stopping realtime recording message
                print("Stopping realtime recording...")
            final_text = self.realtime_pipeline.stop_recording()
            if final_text:
                if False:  # Disable final text message
                    print(f"Final text: {final_text}")
                if False:  # Disable typing final text message
                    print("Typing final text...")
                self.keyboard_typer.type_text(final_text)
                if False:  # Disable text output completed message
                    print("Text output completed!")
            else:
                if False:  # Disable no final text message
                    print("No final text received")
        else:
            # Use traditional batch mode
            self._stop_traditional_recording()
    
    def _start_traditional_recording(self):
        """Start traditional batch recording"""
        success = self.recorder.start_recording()
        
        if success:
            self.recording = True
            if False:  # Disable recording started message
                print("Recording started...")
        else:
            print("Failed to start recording")
    
    def _stop_traditional_recording(self):
        """Stop traditional batch recording and process audio"""
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
                        threshold=self.vad_threshold,
                        aggressiveness=self.vad_aggressiveness,
                        alpha=0.3,  # VAD sensitivity parameter
                        mercy_time_ms=self.config.get("vad_mercy_time_ms", 300)
                    )
                    
                    speech_ratio = vad_result["speech_ratio"]
                    silence_ratio = vad_result["silence_ratio"]
                    
                    print(f"Speech: {speech_ratio*100:.1f}% | Silence: {silence_ratio*100:.1f}%")
                    
                    # Debug: Show detailed VAD information
                    if self.debug_mode and self.debug_vad_details:
                        print(f"Debug VAD Details:")
                        print(f"  - Backend: {self.vad_backend}")
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
        if False:  # Disable cleaning up message
            print("\nCleaning up...")
        
        # Stop any ongoing recording
        if self.recording:
            if self.realtime_mode and self.realtime_pipeline:
                self.realtime_pipeline.stop_recording()
            else:
                self.recorder.stop_recording()
        
        # Clean up realtime pipeline
        if self.realtime_pipeline:
            self.realtime_pipeline.cleanup()
        
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
            if os.path.exists("temp_segments"):
                shutil.rmtree("temp_segments")
                print("Temporary segments cleaned up")
        except Exception as e:
                print(f"Could not clean temp files: {e}")
        
        if False:  # Disable goodbye message
            print("Goodbye!")
    

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper Minimal - Smart Voice Input Assistant")
    parser.add_argument("--config", "-c", 
                       help="Path to custom JSON config file")
    
    args = parser.parse_args()
    
    print("=== Whisper Minimal ===")
    print("Smart Voice Input Assistant")
    print("Press Ctrl+C to exit")
    print()
    
    try:
        app = App(config_file=args.config)
        app.start()
    except Exception as e:
        print(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()