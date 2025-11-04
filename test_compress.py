#!/usr/bin/env python3
"""
Test script for audio compression

Compress WAV file in Target/ folder and save as temporary file for testing
"""

import os
import sys
import numpy as np
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


def check_file_size(file_path: str, max_size_mb: float = 25.0) -> bool:
    """
    Check if file size is within limit
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum size in MB
        
    Returns:
        True if within limit, False otherwise
    """
    file_size = os.path.getsize(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        file_size_mb = file_size / (1024 * 1024)
        print(f"Warning: File size ({file_size_mb:.2f} MB) exceeds limit ({max_size_mb} MB)")
        return False
    
    return True


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
                        return False
            except Exception as e:
                print(f"Error with pydub: {e}")
        
        # Try soundfile (works with WAV, FLAC, etc.)
        if HAS_SOUNDFILE:
            try:
                print("Compressing with soundfile to 16kHz mono, 16-bit WAV...")
                data, sample_rate = sf.read(input_file)
                
                print(f"Original: sample_rate={sample_rate}Hz, channels={data.shape[1] if len(data.shape) > 1 else 1}, dtype={data.dtype}")
                
                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                # Resample to 16kHz if needed
                from scipy import signal
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
                        
                        # Calculate target sample rate to fit 25MB (with 5% margin)
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
                                # For 29.6min audio, need ~7400Hz to be under 25MB
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


def main() -> None:
    """Main entry point"""
    # Setup paths
    target_dir = Path("Target")
    if not target_dir.exists():
        print(f"Error: Target directory not found: {target_dir}")
        sys.exit(1)
    
    # Find WAV file in Target directory
    audio_files = list(target_dir.glob("*.wav")) + list(target_dir.glob("*.WAV"))
    
    if len(audio_files) == 0:
        print(f"Error: No wav file found in {target_dir}")
        sys.exit(1)
    
    if len(audio_files) > 1:
        print(f"Warning: Found {len(audio_files)} wav files, processing only the first one")
    
    input_file = audio_files[0]
    output_file = target_dir / "compressed_test.wav"
    
    # Show original file info
    original_size = os.path.getsize(input_file) / (1024 * 1024)
    print(f"Input file: {input_file.name}")
    print(f"Original size: {original_size:.2f} MB")
    print(f"Output file: {output_file.name}")
    print()
    
    # Compress audio file
    print("Starting compression...")
    success = compress_audio(str(input_file), str(output_file), 24.0)
    
    if success:
        final_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"Compression completed successfully!")
        print(f"Output file: {output_file}")
        print(f"Final size: {final_size:.2f} MB")
        print(f"{'='*60}")
        print("\nYou can now test the compressed file to check audio quality.")
        print("The file will be kept for your review.")
    else:
        print("\nCompression failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

