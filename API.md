# API Documentation

This document describes the public API for Whisper Minimal voice transcription system.

## Core Classes

### `App`

Main application orchestrator managing recording, transcription, and text output.

**Location:** `whisper_minimal.py`

**Methods:**
- `__init__(preset: Optional[str] = None, config_file: Optional[str] = None)` - Initialize application with optional preset or config file
- `start()` - Start the main application loop and register hotkeys

**Configuration:**
- `sample_rate`: Audio sample rate (default: 16000)
- `channels`: Number of audio channels (default: 1)
- `hotkey`: Hotkey combination (default: "ctrl+space")
- `vad_enabled`: Enable VAD preprocessing (default: True)
- `vad_backend`: VAD algorithm ("energy" or "threshold")
- `vad_mercy_time_ms`: Grace period after voice detection in milliseconds (default: 300)
- `realtime_mode`: Enable realtime pipeline mode (default: True)
- `debug_mode`: Enable debug output (default: False)

### `Recorder`

Audio recording using sounddevice library.

**Location:** `recorder.py`

**Methods:**
- `__init__(sample_rate: int = 16000, channels: int = 1)` - Initialize recorder
- `start_recording(callback: Optional[Callable] = None) -> bool` - Start recording, optionally with callback
- `stop_recording() -> Optional[np.ndarray]` - Stop recording and return audio data
- `get_audio_duration(audio_data: np.ndarray) -> float` - Calculate audio duration in seconds

### `Transcriber`

OpenAI Whisper API transcription service.

**Location:** `transcriber.py`

**Methods:**
- `__init__(api_key: str)` - Initialize with OpenAI API key
- `transcribe(audio_file: str) -> Optional[str]` - Transcribe audio file, returns text or None

### `TextCleaner`

GPT-based text cleaning and reorganization.

**Location:** `text_cleaner.py`

**Methods:**
- `__init__(api_key: str)` - Initialize with OpenAI API key
- `clean(text: str) -> Optional[str]` - Clean and reorganize text, returns cleaned text or None

### `KeyboardTyper`

Clipboard copy and paste helper.

**Location:** `keyboard_typer.py`

**Methods:**
- `__init__()` - Initialize clipboard utilities
- `type_text(text: str) -> bool` - Copy text to clipboard and paste with Ctrl+V

## VAD Module

### `run_vad()`

Voice Activity Detection function for audio files.

**Location:** `vad/vad.py`

**Function Signature:**
```python
run_vad(audio_file: str, backend: str = "energy", sample_rate: int = 16000,
        threshold: float = 0.01, aggressiveness: int = 3, alpha: float = 0.5,
        mercy_time_ms: float = 300.0) -> dict
```

**Parameters:**
- `audio_file`: Path to audio file
- `backend`: VAD algorithm ("energy" or "threshold")
- `sample_rate`: Audio sample rate in Hz
- `threshold`: Fixed threshold for threshold VAD
- `aggressiveness`: Aggressiveness level (legacy parameter)
- `alpha`: Sensitivity parameter for energy VAD (0.3-1.0)
- `mercy_time_ms`: Grace period after voice detection in milliseconds

**Returns:**
Dictionary with keys:
- `speech_ratio`: Ratio of speech frames (0.0-1.0)
- `silence_ratio`: Ratio of silence frames (0.0-1.0)
- `vad_mask`: Binary mask array
- `energy_sequence`: Energy values per frame
- `is_voice`: Boolean indicating if voice was detected

### `EnergyVAD`

Energy-based voice activity detector with adaptive threshold.

**Location:** `vad/energy_vad.py`

**Methods:**
- `__init__(sample_rate: int = 16000, frame_length_ms: float = 25.0, hop_length_ms: float = 10.0, alpha: float = 0.7, min_energy_threshold: float = 1e-6)` - Initialize VAD
- `process_audio(audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]` - Process audio, returns VAD mask and energy sequence
- `is_voice_activity(audio_data: np.ndarray) -> Tuple[bool, str]` - Check if audio contains voice, returns (is_voice, reason)
- `reset()` - Reset VAD state

**Parameters:**
- `alpha`: Adaptive threshold sensitivity (0.3-1.0), lower is more sensitive

### `ThresholdVAD`

Simple threshold-based voice activity detector.

**Location:** `vad/threshold_vad.py`

**Methods:**
- `__init__(sample_rate: int = 16000, frame_length_ms: float = 25.0, hop_length_ms: float = 10.0, threshold: float = 0.01)` - Initialize VAD
- `process_audio(audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]` - Process audio, returns VAD mask and energy sequence
- `is_voice_activity(audio_data: np.ndarray) -> Tuple[bool, str]` - Check if audio contains voice, returns (is_voice, reason)
- `reset()` - Reset VAD state (no-op for threshold VAD)

## Realtime Pipeline

### `RealtimePipeline`

Realtime speech transcription pipeline manager.

**Location:** `realtime/pipeline.py`

**Methods:**
- `__init__(config: Dict[str, Any], api_key: str)` - Initialize pipeline with config and API key
- `start_recording() -> bool` - Start recording session
- `cut_and_process_segment()` - Cut current segment and process immediately
- `stop_recording() -> Optional[str]` - Stop recording and return final text
- `cleanup()` - Clean up resources

### `SegmentProcessor`

Processes speech segments detected by VAD.

**Location:** `realtime/processor.py`

**Methods:**
- `__init__(transcriber, text_cleaner, sample_rate: int = 16000, max_queue_size: int = 10, processing_timeout: float = 30.0)` - Initialize processor
- `start()` - Start processing thread
- `stop()` - Stop processing thread
- `add_segment(segment_data: Dict[str, Any]) -> bool` - Add segment to processing queue
- `get_queue_status() -> Dict[str, Any]` - Get queue status and statistics
- `cleanup()` - Clean up resources

### `TextAggregator`

Collects and organizes segment transcription results.

**Location:** `realtime/processor.py`

**Methods:**
- `__init__()` - Initialize aggregator
- `add_segment(text: str, start_time: float, end_time: float)` - Add segment with timestamps
- `add_text_segment(text: str)` - Add text segment (simplified version)
- `get_current_text() -> str` - Get current aggregated text
- `reset()` - Reset aggregator state
- `get_statistics() -> Dict[str, Any]` - Get aggregation statistics

## Usage Examples

### Basic Usage

```python
from whisper_minimal import App

app = App(config_file="config.json")
app.start()
```

### Custom Configuration

```python
app = App()
app.config["vad_backend"] = "energy"
app.config["vad_alpha"] = 0.5
app.config["debug_mode"] = True
app.start()
```

### VAD Analysis

```python
from vad.vad import run_vad

result = run_vad(
    "audio.wav",
    backend="energy",
    alpha=0.5,
    mercy_time_ms=300
)

if result["is_voice"]:
    print(f"Speech ratio: {result['speech_ratio']*100:.1f}%")
```

### Realtime Pipeline

```python
from realtime import RealtimePipeline

config = {
    "sample_rate": 16000,
    "realtime_mode": True,
    "vad_backend": "energy"
}

pipeline = RealtimePipeline(config, api_key="your-api-key")
pipeline.start_recording()
# ... recording logic ...
final_text = pipeline.stop_recording()
```

## Error Handling

All methods that interact with external services (OpenAI API, audio recording) may raise exceptions. Methods typically return `None` or `False` on failure rather than raising exceptions for recoverable errors.

## Thread Safety

The `SegmentProcessor` uses a background thread for processing segments. The `RealtimePipeline` is designed to be thread-safe for recording and processing operations.

## Configuration Files

Configuration is loaded from JSON files with the following structure:

```json
{
  "sample_rate": 16000,
  "channels": 1,
  "vad_enabled": true,
  "vad_backend": "energy",
  "vad_mercy_time_ms": 300,
  "realtime_mode": true,
  "debug_mode": false
}
```

