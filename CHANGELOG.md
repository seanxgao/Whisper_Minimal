# Changelog

All notable changes to this project will be documented in this file.

## 2025-10-23

### VAD System Restructure
- Reorganized VAD system into dedicated `vad/` folder
- Created `vad/vad.py` as main VAD controller
- Implemented two VAD algorithms:
  - **ThresholdVAD**: Simple fixed threshold detection
  - **EnergyVAD**: Statistical adaptive threshold detection
- Unified mercy time mechanism applied to all VAD algorithms

### Configuration System
- Added JSON configuration support via `config.json`
- Moved from hardcoded settings to configurable parameters
- API key handling: automatic environment variable detection with manual input fallback
- Removed unnecessary preset system and configuration files

### Code Cleanup
- Renamed `api_transcriber.py` to `whisper_minimal.py` for consistency
- Removed redundant files and functions
- Simplified VAD interfaces and removed over-engineering
- Consolidated documentation into single README.md

## 2025-10-22

### Pipeline
- Implemented segmented recording pipeline with manual cuts:
  - `Ctrl+Shift+Space` to start/end session (toggle)
  - `Ctrl+Space` to cut current segment and start next
  - Each segment is immediately sent to Whisper for transcription and aggregated
- Console shows intermediate states and per-segment transcription results
- Future direction: automatic dynamic segmentation using VAD-like methods (detect pauses and cut/upload automatically)

### Keyboard Output
- Switched from simulated typing to clipboard-based output with auto-paste (Ctrl+V)
- Avoids issues with Chinese punctuation duplication and missing first character
- Keeps text in clipboard for manual paste if needed

### Project Structure / Portable
- Merged the portable project into the main repo as a build-only folder `portable/`
- `portable/build_portable.py` copies sources from the minimal project into a temp dir and builds a single-file EXE
- Temporary build artifacts are cleaned automatically; `portable/` stays minimal (build script + final EXE)
- `portable/README.md` content consolidated into the root `README.md`
