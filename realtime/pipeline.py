"""
Realtime Pipeline Manager - Coordinates realtime recording, VAD detection, segment transcription and final cleanup
"""

import time
import threading
import numpy as np
from typing import Dict, Any, Optional, Callable
import pyperclip

from .processor import SegmentProcessor, TextAggregator
from recorder import Recorder
from transcriber import Transcriber
from text_cleaner import TextCleaner
from keyboard_typer import KeyboardTyper
from vad.energy_vad import EnergyVAD

class RealtimePipeline:
    """Realtime speech transcription pipeline"""
    
    def __init__(self, config: Dict[str, Any], api_key: str):
        """
        Initialize realtime pipeline
        
        Args:
            config: Configuration dictionary
            api_key: API key
        """
        self.config = config
        self.api_key = api_key
        
        # Initialize components
        self.recorder = Recorder(
            sample_rate=config.get("sample_rate", 16000),
            channels=config.get("channels", 1)
        )
        
        self.transcriber = Transcriber(api_key)
        self.text_cleaner = TextCleaner(api_key)
        self.keyboard_typer = KeyboardTyper()
        
        # Use energy VAD for voice activity detection
        self.energy_vad = EnergyVAD(sample_rate=config.get("sample_rate", 16000), alpha=0.3)
        
        # Audio segmentation parameters
        self.sample_rate = config.get("sample_rate", 16000)
        self.frame_duration = config.get("realtime_frame_duration", 0.1)
        self.silence_threshold = config.get("realtime_silence_threshold", 1.0)
        self.min_speech_duration = config.get("realtime_min_speech_duration", 0.3)
        self.margin = config.get("realtime_margin", 0.2)
        
        # Initialize segment processor
        self.segment_processor = SegmentProcessor(
            transcriber=self.transcriber,
            text_cleaner=self.text_cleaner,
            sample_rate=config.get("sample_rate", 16000),
            max_queue_size=config.get("realtime_max_queue_size", 10),
            processing_timeout=config.get("realtime_processing_timeout", 30.0)
        )
        
        # Initialize text aggregator
        self.text_aggregator = TextAggregator()
        
        # State management
        self.is_recording = False
        self.is_processing = False
        self.session_start_time = None
        
        # Statistics
        self.session_segments = 0
        self.session_text_length = 0
        
        # Setup callback functions
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup callback functions"""
        # Segment processor callbacks
        self.segment_processor.on_text_ready = self._on_text_ready
        self.segment_processor.on_processing_error = self._on_processing_error
        
        # Text aggregator callbacks
        self.text_aggregator.on_final_text_ready = self._on_final_text_ready
    
    def start_recording(self) -> bool:
        """Start recording"""
        if self.is_recording:
            return False
        
        try:
            # Start segment processor
            self.segment_processor.start()
            
            # Reset state
            self.energy_vad.reset()
            self.text_aggregator.reset()
            self.session_start_time = time.time()
            self.session_segments = 0
            self.session_text_length = 0
            
            # Start recording
            success = self.recorder.start_recording(callback=self._audio_callback)
            
            if success:
                self.is_recording = True
                self.is_processing = True
                return True
            else:
                print("[Pipeline] Failed to start recording")
                return False
                
        except Exception as e:
            print(f"[Pipeline] Failed to start recording: {e}")
            return False
    
    def cut_and_process_segment(self):
        """Cut current segment and process immediately"""
        if not self.is_recording:
            return
        
        if self.config.get("debug_mode", False):
            print("[Pipeline] Cutting current segment...")
        
        # Get current audio segment
        segment = self._get_current_segment()
        if segment:
            if self.config.get("debug_mode", False):
                print(f"[Pipeline] Current segment info: duration {segment['duration']:.2f}s")
            
            # VAD detection
            is_voice, reason = self.energy_vad.is_voice_activity(segment["audio_data"])
            if is_voice:
                if self.config.get("debug_mode", False):
                    print(f"[Pipeline] VAD passed: {reason}")
                if not self.config.get("debug_mode", False):
                    print("Transcribing with Whisper API...")
                else:
                    print("[Pipeline] Sending to Whisper API...")
                
                # Process this segment immediately
                self._process_segment_immediately(segment)
            else:
                print(f"[Pipeline] VAD filtered: {reason}")
        else:
            print("[Pipeline] No audio data to process")
    
    def _process_segment_immediately(self, segment):
        """Process audio segment immediately and show results"""
        try:
            # Save audio data to temporary file first
            temp_file = self._save_audio_to_temp_file(segment["audio_data"])
            if temp_file:
                transcription_result = self.transcriber.transcribe(temp_file)
                # Clean up temp file
                import os
                try:
                    os.remove(temp_file)
                except:
                    pass
            else:
                transcription_result = None
            
            if transcription_result and transcription_result.strip():
                text = transcription_result.strip()
                if not self.config.get("debug_mode", False):
                    print("Transcription completed!")
                    print(f"Whisper API: {text}")
                else:
                    print(f"[Pipeline] Transcription result: {text}")
                
                # Add to text aggregator
                self.text_aggregator.add_text_segment(text)
                
                # Show current buffer status
                current_buffer = self.text_aggregator.get_current_text()
                if not self.config.get("debug_mode", False):
                    print(f"[Pipeline] Current buffer: {current_buffer}")
                else:
                    print(f"[Pipeline] Current buffer: {current_buffer}")
                    print("-" * 50)
            else:
                print("[Pipeline] Transcription failed or empty result")
                
        except Exception as e:
            print(f"[Pipeline] Error processing segment: {e}")
    
    def manual_cut(self):
        """Manually cut current segment and send for processing"""
        if not self.is_recording:
            return
        
        print("[Pipeline] Manual segment cut - processing current segment")
        
        # Cut current segment
        # Manual cut - process current audio buffer
        segment = self._get_current_segment()
        if segment:
            # Check VAD before processing
            is_voice, reason = self.energy_vad.is_voice_activity(segment["audio_data"])
            if is_voice:
                self.segment_processor.add_segment(segment)
                print(f"[Pipeline] Segment sent for processing: {segment['duration']:.2f}s")
            else:
                print(f"[Pipeline] Segment filtered by VAD: {reason}")
        else:
            print("[Pipeline] No current segment to cut")
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and return final text"""
        if not self.is_recording:
            return None
        
        try:
            self.is_recording = False
            if self.config.get("debug_mode", False):
                print("[Pipeline] Ending recording session...")
            
            # Process final audio segment
            final_segment = self._get_current_segment()
            if final_segment:
                if self.config.get("debug_mode", False):
                    print(f"[Pipeline] Processing final segment: duration {final_segment['duration']:.2f}s")
                
                # VAD detection
                is_voice, reason = self.energy_vad.is_voice_activity(final_segment["audio_data"])
                if is_voice:
                    if self.config.get("debug_mode", False):
                        print(f"[Pipeline] Final segment VAD passed: {reason}")
                    if self.config.get("debug_mode", False):
                        print("[Pipeline] Sending final segment to Whisper API...")
                    else:
                        pass  # Sending final segment to Whisper API
                    
                    # Process final segment immediately
                    self._process_segment_immediately(final_segment)
                else:
                    print(f"[Pipeline] Final segment VAD filtered: {reason}")
            else:
                print("[Pipeline] No final audio segment")
            
            # Stop recording
            audio_data = self.recorder.stop_recording()
            
            # Wait for all segments to be processed
            self._wait_for_processing_complete()
            
            # Stop segment processor
            self.segment_processor.stop()
            
            # Get current buffered text
            current_text = self.text_aggregator.get_current_text()
            if self.config.get("debug_mode", False):
                print(f"[Pipeline] Final buffer text: {current_text}")
            
            # Send to Cleanup API
            if current_text.strip():
                if not self.config.get("debug_mode", False):
                    print("Starting GPT text cleaning...")
                else:
                    print("[Pipeline] Sending to Cleanup API...")
                final_text = self.text_cleaner.clean(current_text)
                if not self.config.get("debug_mode", False):
                    print(f"Final result: {final_text}")
                else:
                    print(f"[Pipeline] Cleanup result: {final_text}")
            else:
                final_text = current_text
                print("[Pipeline] No text to clean")
            
            self.is_processing = False
            
            # Show session statistics
            self._show_session_statistics()
            return final_text
            
        except Exception as e:
            print(f"[Pipeline] Failed to stop recording: {e}")
            self.is_processing = False
            return None
    
    def _audio_callback(self, indata: np.ndarray):
        """Audio callback function"""
        if not self.is_recording:
            return
        
        # Add audio frame to segmenter
        # Add audio frame to buffer
        self._add_audio_frame(indata.flatten())
    
    def _on_segment_ready(self, segment_data: Dict[str, Any]):
        """Segment ready callback"""
        self.session_segments += 1
        print(f"[Pipeline] Segment ready #{self.session_segments}: {segment_data['duration']:.2f}s")
        
        # Add to processing queue
        success = self.segment_processor.add_segment(segment_data)
        if not success:
            print("[Pipeline] Processing queue full, segment skipped")
    
    def _on_text_ready(self, text_data: Dict[str, Any]):
        """Text ready callback"""
        text = text_data["text"]
        start_time = text_data["start_time"]
        end_time = text_data["end_time"]
        
        print(f"[Pipeline] Segment transcription completed: {text[:50]}...")
        
        # Add to text aggregator
        self.text_aggregator.add_segment(text, start_time, end_time)
        self.session_text_length += len(text)
        
        # Real-time display of current aggregated text
        current_text = self.text_aggregator.get_current_text()
        print(f"[Pipeline] Current text: {current_text}")
    
    def _on_processing_error(self, error: Exception):
        """Processing error callback"""
        print(f"[Pipeline] Processing error: {error}")
    
    def _on_final_text_ready(self, final_text: str):
        """Final text ready callback"""
        print(f"[Pipeline] Final text ready: {final_text}")
        
        # Type text and copy to clipboard
        self.keyboard_typer.type_text(final_text)
        pyperclip.copy(final_text)
        print("[Pipeline] Text typed and copied to clipboard!")
    
    def _wait_for_processing_complete(self, timeout: float = 30.0):
        """Wait for all segments to be processed"""
        start_time = time.time()
        
        while self.segment_processor.is_running:
            queue_status = self.segment_processor.get_queue_status()
            
            if queue_status["queue_size"] == 0:
                # Queue is empty, wait a bit to ensure no new tasks
                time.sleep(0.5)
                if self.segment_processor.get_queue_status()["queue_size"] == 0:
                    break
            
            if time.time() - start_time > timeout:
                print(f"[Pipeline] Timeout waiting for processing to complete ({timeout}s)")
                break
            
            time.sleep(0.1)
    
    def _show_session_statistics(self):
        """Show session statistics"""
        if self.session_start_time is None:
            return
        
        session_duration = time.time() - self.session_start_time
        processor_stats = self.segment_processor.get_queue_status()
        aggregator_stats = self.text_aggregator.get_statistics()
        
        if self.config.get("debug_mode", False):
            print("\n=== Session Statistics ===")
            print(f"Session duration: {session_duration:.1f}s")
            print(f"Speech segments: {self.session_segments}")
            print(f"Current text: {len(self.text_aggregator.get_current_text())} characters")
            print(f"Processor statistics:")
            print(f"  - Total segments: {processor_stats['total_segments']}")
            print(f"  - Processed: {processor_stats['processed_segments']}")
            print(f"  - Failed: {processor_stats['failed_segments']}")
            print(f"  - Skipped: {processor_stats['skipped_segments']}")
            print(f"Aggregator statistics:")
            print(f"  - Segment count: {aggregator_stats['segment_count']}")
            print(f"  - Total duration: {aggregator_stats['total_duration']:.1f}s")
            print("========================")
        else:
            pass  # Session statistics removed for cleaner output
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        processor_stats = self.segment_processor.get_queue_status()
        aggregator_stats = self.text_aggregator.get_statistics()
        
        return {
            "is_recording": self.is_recording,
            "is_processing": self.is_processing,
            "session_duration": time.time() - self.session_start_time if self.session_start_time else 0,
            "processor_stats": processor_stats,
            "aggregator_stats": aggregator_stats,
            "current_text": self.text_aggregator.get_current_text()
        }
    
    def _add_audio_frame(self, audio_data: np.ndarray):
        """Add audio frame to buffer"""
        if not hasattr(self, 'audio_buffer'):
            self.audio_buffer = []
        self.audio_buffer.append(audio_data)
    
    def _get_current_segment(self) -> Optional[Dict[str, Any]]:
        """Get current audio segment from buffer"""
        if not hasattr(self, 'audio_buffer') or not self.audio_buffer:
            return None
        
        # Combine all buffered audio
        audio_data = np.concatenate(self.audio_buffer)
        duration = len(audio_data) / self.sample_rate
        
        # Clear buffer
        self.audio_buffer = []
        
        return {
            "audio_data": audio_data,
            "duration": duration,
            "timestamp": time.time()
        }
    
    def _save_audio_to_temp_file(self, audio_data: np.ndarray) -> Optional[str]:
        """Save audio data to temporary file"""
        if len(audio_data) == 0:
            print(f"[Pipeline] Audio data is empty")
            return None
        
        try:
            import tempfile
            import wave
            import os
            
            # Generate temporary filename
            timestamp = int(time.time() * 1000)
            temp_file = os.path.join(tempfile.gettempdir(), f"whisper_segment_{timestamp}.wav")
            
            # Ensure audio data is int16 format
            if audio_data.dtype != np.int16:
                # Assume input is float32, convert to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data
            
            # Save as WAV file
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            return temp_file
            
        except Exception as e:
            print(f"[Pipeline] Failed to save audio file: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_recording()
        
        self.segment_processor.cleanup()
