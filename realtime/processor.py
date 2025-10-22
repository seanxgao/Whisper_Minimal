"""
Segment Processor - Processes speech segments detected by realtime VAD
"""

import os
import time
import threading
import wave
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from queue import Queue, Empty
import tempfile

class SegmentProcessor:
    """Segment processor responsible for processing speech segments detected by VAD"""
    
    def __init__(self, 
                 transcriber,
                 text_cleaner,
                 sample_rate: int = 16000,
                 max_queue_size: int = 10,
                 processing_timeout: float = 30.0):
        """
        Initialize segment processor
        
        Args:
            transcriber: Transcriber instance
            text_cleaner: Text cleaner instance
            sample_rate: Audio sample rate
            max_queue_size: Maximum queue size
            processing_timeout: Processing timeout (seconds)
        """
        self.transcriber = transcriber
        self.text_cleaner = text_cleaner
        self.sample_rate = sample_rate
        self.max_queue_size = max_queue_size
        self.processing_timeout = processing_timeout
        
        # Queue and threads
        self.segment_queue = Queue(maxsize=max_queue_size)
        self.processing_thread = None
        self.is_running = False
        
        # Statistics
        self.total_segments = 0
        self.processed_segments = 0
        self.failed_segments = 0
        self.skipped_segments = 0
        
        # Callback functions
        self.on_text_ready: Optional[Callable] = None
        self.on_processing_error: Optional[Callable] = None
        
        # Temporary file directory
        self.temp_dir = tempfile.mkdtemp(prefix="whisper_segments_")
        print(f"  - Temp directory: {self.temp_dir}")
        
        print(f"Segment processor initialized:")
        print(f"  - Sample rate: {sample_rate}Hz")
        print(f"  - Max queue size: {max_queue_size}")
        print(f"  - Processing timeout: {processing_timeout}s")
    
    def start(self):
        """Start segment processor"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        print("[SegmentProcessor] Processor started")
    
    def stop(self):
        """Stop segment processor"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        print("[SegmentProcessor] Processor stopped")
    
    def add_segment(self, segment_data: Dict[str, Any]) -> bool:
        """
        Add speech segment to processing queue
        
        Args:
            segment_data: Speech segment data
                {
                    "start_time": float,
                    "end_time": float,
                    "duration": float,
                    "audio_data": np.ndarray
                }
        
        Returns:
            True if added successfully, False if queue is full
        """
        if not self.is_running:
            return False
        
        try:
            self.segment_queue.put_nowait(segment_data)
            self.total_segments += 1
            print(f"[SegmentProcessor] Segment added to queue: {segment_data['duration']:.2f}s")
            return True
        except:
            print(f"[SegmentProcessor] Queue full, skipping segment: {segment_data['duration']:.2f}s")
            self.skipped_segments += 1
            return False
    
    def _processing_loop(self):
        """Processing loop"""
        while self.is_running:
            try:
                # Get segment from queue with timeout
                segment_data = self.segment_queue.get(timeout=1.0)
                
                # Process segment
                self._process_segment(segment_data)
                
                # Mark task as done
                self.segment_queue.task_done()
                
            except Empty:
                # Queue is empty, continue waiting
                continue
            except Exception as e:
                print(f"[SegmentProcessor] Processing loop error: {e}")
                if self.on_processing_error:
                    self.on_processing_error(e)
    
    def _process_segment(self, segment_data: Dict[str, Any]):
        """Process single speech segment"""
        start_time = time.time()
        
        try:
            # Save audio to temporary file
            temp_file = self._save_audio_to_temp(segment_data["audio_data"])
            
            if temp_file is None:
                print(f"[SegmentProcessor] Cannot save audio file")
                self.failed_segments += 1
                return
            
            # Transcribe audio
            print(f"[SegmentProcessor] Starting transcription: {segment_data['duration']:.2f}s")
            text = self.transcriber.transcribe(temp_file)
            
            if text and text.strip():
                print(f"[SegmentProcessor] Transcription completed: {text[:50]}...")
                
                # Send text to callback function
                if self.on_text_ready:
                    self.on_text_ready({
                        "text": text,
                        "start_time": segment_data["start_time"],
                        "end_time": segment_data["end_time"],
                        "duration": segment_data["duration"],
                        "is_final": False  # This is segment result, not final result
                    })
                
                self.processed_segments += 1
            else:
                print(f"[SegmentProcessor] Transcription failed or empty")
                self.failed_segments += 1
            
            # Clean up temporary file
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"[SegmentProcessor] Failed to process segment: {e}")
            self.failed_segments += 1
            if self.on_processing_error:
                self.on_processing_error(e)
        
        finally:
            processing_time = time.time() - start_time
            print(f"[SegmentProcessor] Processing completed, time: {processing_time:.2f}s")
    
    def _save_audio_to_temp(self, audio_data: np.ndarray) -> Optional[str]:
        """Save audio data to temporary file"""
        if len(audio_data) == 0:
            print(f"[SegmentProcessor] Audio data is empty")
            return None
        
        try:
            # Generate temporary filename
            timestamp = int(time.time() * 1000)
            temp_file = os.path.join(self.temp_dir, f"segment_{timestamp}.wav")
            
            print(f"[SegmentProcessor] Saving audio: {len(audio_data)} samples, dtype: {audio_data.dtype}")
            
            # Ensure audio data is int16 format
            if audio_data.dtype != np.int16:
                # Assume input is float32, convert to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                print(f"[SegmentProcessor] Converted to int16: {audio_int16.dtype}")
            else:
                audio_int16 = audio_data
            
            # Save as WAV file
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            print(f"[SegmentProcessor] Audio saved to: {temp_file}")
            return temp_file
            
        except Exception as e:
            print(f"[SegmentProcessor] Failed to save audio file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status"""
        return {
            "queue_size": self.segment_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "is_running": self.is_running,
            "total_segments": self.total_segments,
            "processed_segments": self.processed_segments,
            "failed_segments": self.failed_segments,
            "skipped_segments": self.skipped_segments
        }
    
    def clear_queue(self):
        """Clear queue"""
        while not self.segment_queue.empty():
            try:
                self.segment_queue.get_nowait()
                self.segment_queue.task_done()
            except Empty:
                break
        print("[SegmentProcessor] Queue cleared")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        
        # Clean up temporary files
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"[SegmentProcessor] Temp directory cleaned: {self.temp_dir}")
        except Exception as e:
            print(f"[SegmentProcessor] Failed to clean temp directory: {e}")


class TextAggregator:
    """Text aggregator for collecting and organizing segment transcription results"""
    
    def __init__(self):
        """Initialize text aggregator"""
        self.segments: List[Dict[str, Any]] = []
        self.final_text = ""
        
        # Callback functions
        self.on_final_text_ready: Optional[Callable] = None
    
    def add_segment(self, segment_text: str, start_time: float, end_time: float):
        """Add segment text"""
        self.segments.append({
            "text": segment_text,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time
        })
        
        print(f"[TextAggregator] Added segment: {segment_text[:30]}... ({start_time:.2f}s - {end_time:.2f}s)")
    
    def add_text_segment(self, text: str):
        """Add text segment (simplified version for immediate processing)"""
        import time
        current_time = time.time()
        self.segments.append({
            "text": text,
            "start_time": current_time,
            "end_time": current_time,
            "duration": 0.0
        })
        
        print(f"[TextAggregator] Added text segment: {text[:50]}...")
    
    def get_current_text(self) -> str:
        """Get current aggregated text"""
        return " ".join([seg["text"] for seg in self.segments])
    
    def finalize_and_clean(self, text_cleaner) -> str:
        """Finalize and clean text"""
        if not self.segments:
            return ""
        
        # Aggregate all segment texts
        self.final_text = self.get_current_text()
        
        print(f"[TextAggregator] Starting final cleanup: {self.final_text[:100]}...")
        
        # Use GPT to clean text
        cleaned_text = text_cleaner.clean(self.final_text)
        
        if cleaned_text:
            print(f"[TextAggregator] Final cleanup completed")
            print(f"Original text: {self.final_text}")
            print(f"Cleaned text: {cleaned_text}")
            
            if self.on_final_text_ready:
                self.on_final_text_ready(cleaned_text)
            
            return cleaned_text
        else:
            print(f"[TextAggregator] Cleanup failed, using original text")
            return self.final_text
    
    def reset(self):
        """Reset aggregator"""
        self.segments.clear()
        self.final_text = ""
        print("[TextAggregator] Reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        total_duration = sum(seg["duration"] for seg in self.segments)
        
        return {
            "segment_count": len(self.segments),
            "total_duration": total_duration,
            "current_text_length": len(self.get_current_text()),
            "final_text_length": len(self.final_text)
        }
