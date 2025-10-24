# Whisper Minimal - Smart Voice Input Assistant

A voice recording and transcription tool with GPT text cleaning and streamlined segmented recording pipeline.

## Features

### Core Features
- **Segmented Recording (Manual)**: Ctrl+Shift+Space to start/end session; Ctrl+Space to cut current segment
- **VAD Options**: 
  - **Energy VAD**: Adaptive threshold based on environment noise (default)
  - **Threshold VAD**: Simple fixed threshold detection
- **Unified Mercy Time**: Configurable grace period (300ms default) applied to all VAD algorithms after detection
- **Whisper Transcription**: Uses OpenAI Whisper API for accurate speech-to-text
- **GPT Text Cleanup**: Final cleanup via GPT-based API
- **Clipboard + Auto Paste**: Final text is copied to clipboard and pasted via Ctrl+V
- **Preset Configurations**: Simple, Professional, and Fast presets for different use cases
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
- If you have the API key in environment variables, the program will use it automatically
- If not, the program will prompt you to enter it manually (not stored permanently)

### 3. Run the Program
```bash
python whisper_minimal.py
```

Or with custom configuration:
```bash
python whisper_minimal.py --config config.json
```

## Portable Version

The `portable/` folder is a build-only wrapper that packages the minimal project into a standalone EXE.

### Build Portable Executable
```bash
cd portable

# Build portable executable
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
1. Run: `python whisper_minimal.py`
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
- Analyze audio quality using Energy-based VAD (Voice Activity Detection)
- Apply 300ms mercy time to avoid cutting off speech
- Skip API calls if insufficient speech is detected (saves costs)
- Transcribe speech using OpenAI Whisper API
- Reorganize text using GPT-4o-mini for better clarity and logic
- Type the improved text directly into the active window
- Copy the text to clipboard as backup
- Display processing steps and statistics in the console

## VAD (Voice Activity Detection)

The system uses a simple energy-based VAD algorithm:

### Algorithm
- **Frame Processing**: 25ms frames with 10ms hop length
- **Energy Calculation**: Mean squared energy per frame
- **Adaptive Threshold**: `threshold = mean_energy + alpha * std_energy`
- **Mercy Time**: 300ms grace period after voice detection

### Parameters
- **Alpha**: Controls sensitivity (0.3-1.0)
  - 0.3: More sensitive (Professional preset)
  - 0.5: Balanced (Simple preset)
  - 0.7: Less sensitive (Fast preset)

## Configuration

Edit `config.json` to change settings. Key options:
- **vad_backend**: VAD algorithm ("energy" or "threshold")
- **vad_threshold**: Fixed threshold for threshold VAD (default: 0.01)
- **vad_mercy_time_ms**: Mercy time applied to all VAD algorithms (milliseconds, default: 300)
- **debug_mode**: Enable debug output

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
- `whisper_minimal.py` - Main application orchestrating all components
- `config_utils.py` - Configuration management with sensible defaults

### Core Modules
- `recorder.py` - Audio recording using sounddevice
- `transcriber.py` - OpenAI Whisper API transcription
- `text_cleaner.py` - GPT-4o-mini text cleaning and reorganization
- `keyboard_typer.py` - Outputs final text by copying to clipboard and auto-pasting
- `vad/` - Voice Activity Detection module
- `realtime/` - Realtime pipeline modules
  - `processor.py` - Segment processor and text aggregator
  - `pipeline.py` - Realtime pipeline manager

### Portable Version
- `portable/` - Portable executable build system
  - `build_portable.py` - Builds portable executable from minimal codebase
  - `dist/WhisperPortable.exe` - Generated portable executable

**Note**: The portable version automatically copies all necessary files from minimal during build, including the `realtime/` module and `vad/` directory, ensuring zero code duplication. The portable directory itself contains no source code, only build tools and the generated executable.

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