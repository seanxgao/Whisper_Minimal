#!/usr/bin/env python3
"""
Batch audio transcription script

Process audio files in Target/ folder:
1. Find WAV file in Target directory
2. Transcribe entire audio file with Whisper API
3. Save result to Target/output.txt
"""

import os
import sys
import wave
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

from transcriber import Transcriber
from vad.vad import run_vad


def convert_to_wav(input_file: str, output_file: str) -> bool:
    """
    Convert audio file to WAV format
    
    Args:
        input_file: Input audio file path
        output_file: Output WAV file path
        
    Returns:
        True if successful, False otherwise
    """
    # Try pydub first (works with m4a, mp3, etc.)
    if HAS_PYDUB:
        try:
            audio = AudioSegment.from_file(input_file)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            audio.export(output_file, format="wav")
            return True
        except Exception as e:
            print(f"Error converting with pydub: {e}")
    
    # Try soundfile (works with wav, flac, etc.)
    if HAS_SOUNDFILE:
        try:
            import numpy as np
            data, sample_rate = sf.read(input_file)
            
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            if data.dtype != np.int16:
                data = (data * 32767).astype(np.int16)
            
            sf.write(output_file, data, sample_rate, format='WAV', subtype='PCM_16')
            return True
        except Exception as e:
            print(f"Error converting with soundfile: {e}")
    
    # Try ffmpeg command line as last resort
    try:
        import subprocess
        result = subprocess.run(
            ['ffmpeg', '-i', input_file, '-acodec', 'pcm_s16le', 
             '-ar', '16000', '-ac', '1', '-y', output_file],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and os.path.exists(output_file):
            return True
    except Exception as e:
        print(f"Error converting with ffmpeg: {e}")
    
    print("No suitable conversion method available")
    if not HAS_PYDUB:
        print("Please install pydub: pip install pydub")
        print("Or install ffmpeg for command-line conversion")
    return False


def check_file_size(file_path: str, max_size_mb: float = 25.0) -> bool:
    """
    Check if file size is within Whisper API limit
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum size in MB (default 25MB for Whisper API)
        
    Returns:
        True if within limit, False otherwise
    """
    file_size = os.path.getsize(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        file_size_mb = file_size / (1024 * 1024)
        print(f"Warning: File size ({file_size_mb:.2f} MB) exceeds API limit ({max_size_mb} MB)")
        return False
    
    return True


def find_speech_segments_from_vad(vad_mask: np.ndarray, sample_rate: int, 
                                  hop_length_ms: float = 10.0) -> List[Tuple[float, float]]:
    """
    Find continuous speech segments from VAD mask
    
    Args:
        vad_mask: Binary VAD mask (1.0 for speech, 0.0 for silence)
        sample_rate: Audio sample rate
        hop_length_ms: Hop length in milliseconds
        
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    if len(vad_mask) == 0:
        return []
    
    segments = []
    in_speech = False
    start_idx = None
    
    for i, is_speech in enumerate(vad_mask):
        if is_speech > 0.5:  # Speech frame
            if not in_speech:
                in_speech = True
                start_idx = i
        else:  # Silence frame
            if in_speech:
                # End of speech segment
                end_idx = i
                start_time = start_idx * hop_length_ms / 1000.0
                end_time = end_idx * hop_length_ms / 1000.0
                segments.append((start_time, end_time))
                in_speech = False
                start_idx = None
    
    # Handle segment that extends to end of audio
    if in_speech and start_idx is not None:
        end_idx = len(vad_mask)
        start_time = start_idx * hop_length_ms / 1000.0
        end_time = end_idx * hop_length_ms / 1000.0
        segments.append((start_time, end_time))
    
    return segments


def remove_silence_with_vad(input_file: str, output_file: str) -> bool:
    """
    Remove silence from audio file using VAD, keeping only speech segments
    
    Args:
        input_file: Input audio file path
        output_file: Output audio file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import wave
        import numpy as np
        
        # Run VAD to detect speech segments
        print("Running VAD to detect speech segments...")
        vad_result = run_vad(
            input_file,
            backend="energy",
            sample_rate=16000,
            alpha=0.5,
            mercy_time_ms=500.0
        )
        
        vad_mask = vad_result["vad_mask"]
        speech_ratio = vad_result["speech_ratio"]
        
        print(f"Speech ratio: {speech_ratio*100:.1f}%")
        
        # Find speech segments
        segments = find_speech_segments_from_vad(vad_mask, sample_rate=16000, hop_length_ms=10.0)
        print(f"Found {len(segments)} speech segments")
        
        if len(segments) == 0:
            print("No speech segments detected")
            return False
        
        # Read original audio
        with wave.open(input_file, 'rb') as wav_in:
            sample_rate = wav_in.getframerate()
            n_channels = wav_in.getnchannels()
            sampwidth = wav_in.getsampwidth()
            frames = wav_in.readframes(wav_in.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Extract and concatenate all speech segments
            speech_audio = []
            total_speech_time = 0.0
            
            for start_time, end_time in segments:
                start_frame = int(start_time * sample_rate)
                end_frame = int(end_time * sample_rate)
                segment_audio = audio_data[start_frame:end_frame]
                speech_audio.append(segment_audio)
                total_speech_time += (end_time - start_time)
            
            # Concatenate all segments
            if len(speech_audio) > 0:
                merged_audio = np.concatenate(speech_audio)
                
                # Write merged audio to output file
                with wave.open(output_file, 'wb') as wav_out:
                    wav_out.setnchannels(n_channels)
                    wav_out.setsampwidth(sampwidth)
                    wav_out.setframerate(sample_rate)
                    wav_out.writeframes(merged_audio.tobytes())
                
                original_size = os.path.getsize(input_file) / (1024 * 1024)
                new_size = os.path.getsize(output_file) / (1024 * 1024)
                reduction = (1 - new_size / original_size) * 100
                
                print(f"Silence removed: {original_size:.2f} MB -> {new_size:.2f} MB ({reduction:.1f}% reduction)")
                print(f"Original duration: {len(audio_data)/sample_rate:.1f}s, Speech duration: {total_speech_time:.1f}s")
                
                return True
        
        return False
        
    except Exception as e:
        print(f"Error removing silence: {e}")
        return False


def compress_audio(input_file: str, output_file: str, target_size_mb: float = 24.0) -> bool:
    """
    Compress audio file to reduce size
    
    Args:
        input_file: Input audio file path
        output_file: Output compressed file path
        target_size_mb: Target file size in MB
        
    Returns:
        True if successful, False otherwise
    """
    try:
        original_size = os.path.getsize(input_file) / (1024 * 1024)
        
        # Try pydub first (works with various formats)
        if HAS_PYDUB:
            try:
                print("Compressing with pydub to 16kHz mono, 16-bit WAV...")
                audio = AudioSegment.from_file(input_file)
                audio = audio.set_channels(1)  # Mono
                audio = audio.set_frame_rate(16000)  # 16kHz
                audio = audio.set_sample_width(2)  # 16-bit
                audio.export(output_file, format="wav")
                
                if os.path.exists(output_file):
                    compressed_size = os.path.getsize(output_file)
                    new_size = compressed_size / (1024 * 1024)
                    reduction = (1 - new_size / original_size) * 100
                    
                    print(f"Compression result: {original_size:.2f} MB -> {new_size:.2f} MB ({reduction:.1f}% reduction)")
                    
                    if check_file_size(output_file, 25.0):
                        return True
                    else:
                        print("File still too large after basic compression")
                        # Continue to soundfile method for further compression
            except Exception as e:
                print(f"Error with pydub: {e}")
        
        # Try soundfile (works with WAV, FLAC, etc.)
        if HAS_SOUNDFILE:
            try:
                import numpy as np
                from scipy import signal
                
                print("Compressing with soundfile to 16kHz mono, 16-bit WAV...")
                data, sample_rate = sf.read(input_file)
                
                print(f"Original: sample_rate={sample_rate}Hz, channels={data.shape[1] if len(data.shape) > 1 else 1}, dtype={data.dtype}")
                
                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    print(f"Resampling from {sample_rate}Hz to 16000Hz...")
                    num_samples = int(len(data) * 16000 / sample_rate)
                    data = signal.resample(data, num_samples)
                    sample_rate = 16000
                
                # Normalize and convert to int16
                if data.dtype != np.int16:
                    # Normalize to [-1, 1] range first
                    max_val = np.max(np.abs(data))
                    if max_val > 0:
                        data = data / max_val
                    # Convert to int16 (ensure values are in [-1, 1] range)
                    data = np.clip(data, -1.0, 1.0)
                    data = (data * 32767).astype(np.int16)
                
                # Save as WAV
                print("Saving compressed WAV file...")
                sf.write(output_file, data, 16000, format='WAV', subtype='PCM_16')
                
                if os.path.exists(output_file):
                    compressed_size = os.path.getsize(output_file)
                    new_size = compressed_size / (1024 * 1024)
                    reduction = (1 - new_size / original_size) * 100
                    
                    print(f"Compression result: {original_size:.2f} MB -> {new_size:.2f} MB ({reduction:.1f}% reduction)")
                    
                    if check_file_size(output_file, 25.0):
                        return True
                    else:
                        print(f"File still too large ({new_size:.2f} MB), trying aggressive compression...")
                        
                        # Calculate target sample rate to fit 25MB (with 10% margin)
                        duration_sec = len(data) / 16000
                        target_bytes = 24.0 * 1024 * 1024  # 24MB target for safety margin
                        bytes_per_sec_needed = target_bytes / duration_sec
                        # 16-bit mono = 2 bytes per sample
                        samples_per_sec_needed = bytes_per_sec_needed / 2
                        
                        if samples_per_sec_needed < 16000:
                            # Use 10% safety margin to ensure under 25MB
                            target_sample_rate = max(8000, int(samples_per_sec_needed * 0.90))
                            # Ensure we're under 25MB by further reducing if needed
                            if target_sample_rate >= 8000:
                                # Reduce by additional 5% if at 8kHz
                                target_sample_rate = int(target_sample_rate * 0.92)
                            print(f"Calculated target sample rate: {target_sample_rate}Hz (target: <25MB)")
                            
                            if target_sample_rate < 16000:
                                print(f"Resampling to {target_sample_rate}Hz...")
                                data_compressed = signal.resample(data.astype(np.float32), 
                                                                  int(len(data) * target_sample_rate / 16000))
                                data_compressed = (data_compressed / 32767.0 * 32767).astype(np.int16)
                                
                                sf.write(output_file, data_compressed, target_sample_rate, 
                                        format='WAV', subtype='PCM_16')
                                
                                if os.path.exists(output_file):
                                    compressed_size = os.path.getsize(output_file)
                                    new_size = compressed_size / (1024 * 1024)
                                    reduction = (1 - new_size / original_size) * 100
                                    print(f"{target_sample_rate}Hz compression: {new_size:.2f} MB ({reduction:.1f}% reduction)")
                                    
                                    if check_file_size(output_file, 25.0):
                                        return True
                        
                        return False
            except Exception as e:
                print(f"Error with soundfile: {e}")
                import traceback
                traceback.print_exc()
        
        # Try ffmpeg as last resort
        try:
            import subprocess
            print("Trying ffmpeg compression...")
            result = subprocess.run(
                ['ffmpeg', '-i', input_file, '-acodec', 'pcm_s16le', 
                 '-ar', '16000', '-ac', '1', '-y', output_file],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and os.path.exists(output_file):
                compressed_size = os.path.getsize(output_file)
                new_size = compressed_size / (1024 * 1024)
                reduction = (1 - new_size / original_size) * 100
                print(f"Compression result: {original_size:.2f} MB -> {new_size:.2f} MB ({reduction:.1f}% reduction)")
                return check_file_size(output_file, 25.0)
        except Exception as e:
            print(f"Error with ffmpeg: {e}")
        
        print("No suitable compression method available")
        print("Please install pydub: pip install pydub")
        print("Or install soundfile: pip install soundfile")
        return False
        
    except Exception as e:
        print(f"Error compressing audio: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_audio_file(audio_file: str, transcriber: Transcriber, 
                       output_dir: str) -> List[str]:
    """
    Process audio file with transcription (with timestamps)
    
    Args:
        audio_file: Path to audio file
        transcriber: Transcriber instance
        output_dir: Directory for temporary files
        
    Returns:
        List with single transcribed text with timestamps
    """
    print(f"Processing: {audio_file}")
    
    # Check file size and compress if needed
    # Note: VAD silence removal is disabled for now (see remove_silence_with_vad function)
    # Can be enabled in the future when VAD quality improves
    original_audio = audio_file
    temp_dir = Path(output_dir)
    temp_dir.mkdir(exist_ok=True)
    
    if not check_file_size(audio_file, 25.0):
        print("File exceeds 25MB limit, compressing audio file...")
        compressed_file = temp_dir / f"compressed_{Path(audio_file).name}"
        
        if compress_audio(audio_file, str(compressed_file), 24.0):
            audio_file = str(compressed_file)
        else:
            print("Error: Failed to compress audio file")
            print("Please compress the audio file manually or split it into smaller segments")
            return []
        
        # Future: VAD-based silence removal can be enabled here when VAD quality improves
        # if not check_file_size(audio_file, 25.0):
        #     print("File still too large, removing silence with VAD...")
        #     vad_processed_file = temp_dir / f"vad_processed_{Path(audio_file).name}"
        #     if remove_silence_with_vad(audio_file, str(vad_processed_file)):
        #         if check_file_size(str(vad_processed_file), 25.0):
        #             audio_file = str(vad_processed_file)
    
    audio_path = Path(audio_file)
    wav_file = audio_file
    
    # Convert to WAV if needed
    if audio_path.suffix.lower() != '.wav':
        if not HAS_PYDUB and not HAS_SOUNDFILE:
            print(f"Error: {audio_path.suffix} format requires conversion library")
            print("Please install pydub: pip install pydub")
            print("Or install soundfile: pip install soundfile")
            return []
        
        print(f"Converting {audio_path.suffix} to WAV...")
        temp_wav = Path(output_dir) / f"{audio_path.stem}.wav"
        if not convert_to_wav(audio_file, str(temp_wav)):
            print("Failed to convert audio file")
            return []
        wav_file = str(temp_wav)
    
    # Transcribe entire audio file with timestamps
    print("Transcribing entire audio file with timestamps...")
    text = transcriber.transcribe(wav_file, with_timestamps=True)
    
    transcriptions = []
    if text and text.strip():
        transcriptions.append(text.strip())
        print(f"Transcription completed: {len(text)} characters")
    else:
        print("Transcription failed or empty")
    
    # Clean up temporary files
    original_audio = Path(audio_file).name
    if wav_file != audio_file:
        try:
            os.remove(wav_file)
        except Exception:
            pass
    
    # Clean up compressed file if it was created
    if str(audio_file).startswith(str(Path(output_dir))):
        try:
            os.remove(audio_file)
        except Exception:
            pass
    
    return transcriptions


def get_api_key() -> Optional[str]:
    """
    Get API key from environment or user input
    
    Returns:
        API key string or None if not provided
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            print("No API key provided.")
            return None
    return api_key


def find_audio_file(target_dir: Path) -> Optional[Path]:
    """
    Find WAV file in target directory
    
    Args:
        target_dir: Target directory path
        
    Returns:
        Path to audio file or None if not found
    """
    audio_files = list(target_dir.glob("*.wav")) + list(target_dir.glob("*.WAV"))
    
    if len(audio_files) == 0:
        print(f"Error: No wav file found in {target_dir}")
        return None
    
    if len(audio_files) > 1:
        print(f"Warning: Found {len(audio_files)} wav files, processing only the first one")
    
    return audio_files[0]


def save_results(output_file: Path, transcriptions: List[str]) -> None:
    """
    Save transcription results to file
    
    Args:
        output_file: Output file path
        transcriptions: List of transcribed texts
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in transcriptions:
            f.write(f"{text}\n")


def main() -> None:
    """Main entry point"""
    # Get API key
    api_key = get_api_key()
    if not api_key:
        sys.exit(1)
    
    # Setup paths
    target_dir = Path("Target")
    if not target_dir.exists():
        print(f"Error: Target directory not found: {target_dir}")
        sys.exit(1)
    
    # Find audio file
    audio_file = find_audio_file(target_dir)
    if not audio_file:
        sys.exit(1)
    
    print(f"Processing: {audio_file.name}")
    
    # Initialize transcriber
    transcriber = Transcriber(api_key)
    
    # Create temporary directory
    temp_dir = target_dir / "temp_segments"
    temp_dir.mkdir(exist_ok=True)
    
    # Process audio file
    print(f"\n{'='*60}")
    print(f"Processing: {audio_file.name}")
    print(f"{'='*60}")
    
    transcriptions = process_audio_file(str(audio_file), transcriber, str(temp_dir))
    
    print(f"Completed: {audio_file.name} ({len(transcriptions)} segments)")
    
    # Save results
    output_file = target_dir / "output.txt"
    save_results(output_file, transcriptions)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total segments transcribed: {len(transcriptions)}")
    print(f"{'='*60}")
    
    # Clean up temporary directory
    try:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
