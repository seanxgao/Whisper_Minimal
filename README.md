# Whisper Minimal - Smart Voice Input Assistant

A voice recording and transcription tool with GPT text cleaning and automatic keyboard input.

## Features

- **Hotkey Recording**: Press Ctrl+Space to start/stop recording
- **VAD Preprocessing**: Simple energy-based Voice Activity Detection to filter out silent audio
- **Whisper Transcription**: Uses OpenAI Whisper API for accurate speech-to-text
- **GPT Text Reorganization**: Intelligently reorganizes and improves text clarity
- **Keyboard Input**: Types cleaned text directly into active window
- **Clipboard Backup**: Also copies results to clipboard
- **Cost Optimization**: Skips API calls for low-quality audio
- **Statistics Tracking**: Monitor API usage and cost savings
- **Background Operation**: Runs in background without interfering with other apps

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup API Key
Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Alternatively, create a file with your API key and specify the path in the code.

### 3. Run the Program
```bash
python api_transcriber.py
```

## Usage

1. **Start the program**: Run `python api_transcriber.py`
2. **Start recording**: Press `Ctrl+Space` once to start recording
3. **Stop recording**: Press `Ctrl+Space` again to stop recording
4. **Automatic processing**: 
   - VAD analysis detects speech vs silence
   - Audio is transcribed using Whisper API (if sufficient speech detected)
   - Text is reorganized using GPT-4o-mini for better clarity
   - Improved text is typed directly into the active window
   - Text is also copied to clipboard as backup
5. **Exit program**: Press `Ctrl+C`

**Note**: No need to hold keys! Just press Ctrl+Space once to start, press again to stop. The program runs in the background and won't interfere with other applications.

## Output

The program will:
- Analyze audio quality using VAD (Voice Activity Detection)
- Skip API calls if insufficient speech is detected (saves costs)
- Transcribe speech using OpenAI Whisper API
- Reorganize text using GPT-4o-mini for better clarity and logic
- Type the improved text directly into the active window
- Copy the text to clipboard as backup
- Display processing steps and statistics in the console

## Configuration

Edit `config.json` to customize VAD settings:

```json
{
  "vad_enabled": true,
  "vad_backend": "webrtc",
  "vad_threshold": 0.5,
  "min_speech_ratio": 0.1,
  "vad_aggressiveness": 2
}
```

### VAD Settings:
- `vad_enabled`: Enable/disable VAD preprocessing
- `vad_backend`: "webrtc" (lightweight) or "silero" (more accurate)
- `min_speech_ratio`: Minimum speech percentage to process (0.1 = 10%)
- `vad_aggressiveness`: WebRTC VAD sensitivity (0-3, higher = more aggressive)

## Troubleshooting

### Audio Issues
- Check microphone permissions
- Verify default audio device settings
- Ensure microphone is connected and working

### Hotkey Issues
- Make sure no other programs are using Ctrl+Space
- Try running as administrator if needed

## Files

### Core Application
- `api_transcriber.py` - Main application orchestrating all components
- `config.json` - Configuration file with VAD settings

### Modules
- `config_utils.py` - Configuration loading and management
- `recorder.py` - Audio recording using sounddevice
- `transcriber.py` - OpenAI Whisper API transcription
- `text_cleaner.py` - GPT-4o-mini text cleaning and reorganization
- `keyboard_typer.py` - Cross-platform keyboard input simulation

### VAD System
- `vad/` - VAD module directory
  - `segmenter.py` - Main VAD interface
  - `backends.py` - VAD backend implementations (Simple, Advanced, WebRTC, Silero)
  - `simple_vad.py` - Simple energy-based VAD implementation
  - `ot_vad_smoothing.py` - OT smoothing placeholder
  - `logger.py` - VAD logging utilities

### Documentation
- `requirements.txt` - Dependencies
- `README.md` - This documentation

## Dependencies

- `sounddevice` - Audio recording
- `numpy` - Audio processing
- `openai` - Whisper and GPT API access
- `keyboard` - Hotkey detection
- `pyperclip` - Clipboard operations
- `pynput` - Keyboard simulation
- `requests` - HTTP requests
- `webrtcvad` - Voice Activity Detection (optional, fallback)
- `silero-vad` - Alternative VAD (optional, fallback, requires torch)

## Notes

This is a minimal, elegant, modular voice input assistant that combines:
- **Modular architecture** with separate components for easy maintenance
- **VAD preprocessing** to filter out silent audio and reduce API costs
- **OpenAI Whisper** for accurate speech-to-text transcription
- **GPT-4o-mini** for intelligent text reorganization and improvement
- **Automatic keyboard input** for seamless text entry
- **Professional code structure** with clear separation of concerns

The VAD system helps optimize costs by skipping API calls for low-quality audio while maintaining high transcription accuracy for actual speech content.