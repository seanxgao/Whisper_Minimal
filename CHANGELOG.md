# Changelog

All notable changes to this project will be documented in this file.

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
