# Whisper Minimal - Smart Voice Input Assistant

A voice recording and transcription tool with GPT text cleaning and streamlined segmented recording pipeline.

## Features

### Core Features
- **Segmented Recording (Manual)**: Ctrl+Shift+Space to start/end session; Ctrl+Space to cut current segment
- **Unified VAD (Consolidated)**: Single `unified_vad.py` used across modes
- **Whisper Transcription**: Uses OpenAI Whisper API for accurate speech-to-text
- **GPT Text Cleanup**: Final cleanup via GPT-based API
- **Clipboard + Auto Paste**: Final text is copied to clipboard and pasted via Ctrl+V
- **Zero Config Defaults**: Sensible defaults without `config.json` or advanced config
- **Background Operation**: Runs in background without interfering with other apps

### Segmented Pipeline (Current) and Future Direction
- **Manual Segmentation (Current)**: You control segments with hotkeys; each cut is sent immediately for transcription and aggregation
- **Realtime Feedback**: Console prints intermediate states and transcription results for each segment
- **Future (Planned)**: Automatic dynamic segmentation using VAD-like methods—detect pauses and cut/upload automatically

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

## Portable Version

The `portable/` folder is a build-only wrapper that packages the minimal project into a standalone EXE.

### Build Portable Executable
```bash
cd portable
python build_portable.py
```

This creates `portable/dist/WhisperPortable.exe` — a single-file executable.

### Run Portable Version
1. Navigate to `portable/dist/`
2. Run `WhisperPortable.exe`
3. Enter your API key when prompted
4. Use the same hotkeys as the main version (Segmented Recording Mode)

Note: The portable builder copies sources from the minimal project at build time to avoid code duplication.

## Usage

### Segmented Recording Mode
1. Run: `python api_transcriber.py`
2. Start session: `Ctrl+Shift+Space`
3. Cut current segment and start next: `Ctrl+Space`
4. End session: `Ctrl+Shift+Space` (toggle)
5. Final processing: segments aggregated → GPT cleanup → copied to clipboard and pasted (Ctrl+V)
6. Exit: `Ctrl+C`

### Traditional Mode (Optional)
Basic start/stop recording with batch transcription and cleanup.

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

The application uses sensible defaults and doesn't require a configuration file. Advanced configuration has been removed to simplify usage.

## Troubleshooting

### Audio Issues
- Check microphone permissions
- Verify default audio device settings
- Ensure microphone is connected and working

### Hotkey Issues
- Make sure no other programs are using Ctrl+Space
- Try running as administrator if needed

## Files

### Core Application (Minimal)
- `api_transcriber.py` - Main application orchestrating all components
- `config_utils.py` - Configuration management with sensible defaults

### Core Modules
- `recorder.py` - Audio recording using sounddevice
- `transcriber.py` - OpenAI Whisper API transcription
- `text_cleaner.py` - GPT-4o-mini text cleaning and reorganization
- `keyboard_typer.py` - Outputs final text by copying to clipboard and auto-pasting
- `unified_vad.py` - Unified Voice Activity Detection for all modes
- `realtime/` - Realtime pipeline modules
  - `processor.py` - Segment processor and text aggregator
  - `pipeline.py` - Realtime pipeline manager

### Portable Version
- `portable/` - Portable executable build system
  - `build_portable.py` - Builds portable executable from minimal codebase
  - `dist/WhisperPortable.exe` - Generated portable executable

**Note**: The portable version automatically copies all necessary files from minimal during build, including the `realtime/` module and `unified_vad.py`, ensuring zero code duplication. The portable directory itself contains no source code, only build tools and the generated executable.

### Documentation
- `requirements.txt` - Dependencies
- `README.md` - This documentation
- `CHANGELOG.md` - Changelog and feature change records

## Dependencies

- `sounddevice` - Audio recording
- `numpy` - Audio processing
- `openai` - Whisper and GPT API access
- `keyboard` - Hotkey detection
- `pyperclip` - Clipboard operations
- `pyperclip` - Clipboard operations
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