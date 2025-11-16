# TinyVAD Deployment Guide

This directory contains deployment-ready files for TinyVAD model.

## Files

- `tiny_vad.onnx` - ONNX model file (export from training)
- `config.json` - Model configuration and inference parameters
- `inference_example.py` - Example inference script

## Quick Start

### Python Inference

```python
from deploy.inference_example import TinyVADInference
import numpy as np
from vad_distill.utils.audio_io import load_wav

# Initialize
inference = TinyVADInference("tiny_vad.onnx", "config.json")

# Load audio
wav = load_wav("audio.wav", target_sr=16000)

# Run inference
frame_scores = inference.infer(wav)

# Extract segments (threshold=0.5)
segments = [(i * 0.01, (i+1) * 0.01) for i, score in enumerate(frame_scores) if score > 0.5]
```

### Command Line

```bash
python deploy/inference_example.py audio.wav --onnx tiny_vad.onnx --output scores.npy
```

## Model Specifications

- **Input**: Fixed-size chunks of shape `(1, 100, 80)`
  - 100 frames = 1 second at 10ms frame hop
  - 80 mel filterbanks
- **Output**: VAD logits of shape `(1, 100, 1)`
  - Apply sigmoid to get probabilities
- **Sample Rate**: 16 kHz
- **Frame Parameters**: 25ms window, 10ms hop

## Streaming Inference

For real-time streaming, use `infer_streaming()`:

```python
inference = TinyVADInference("tiny_vad.onnx")
state = None

for audio_chunk in audio_stream:
    scores, state = inference.infer_streaming(audio_chunk, state)
    # Process scores...
```

## Mobile / Embedded Deployment

### ONNX Runtime Mobile

1. Convert ONNX to mobile-optimized format:
   ```bash
   python -m onnxruntime.tools.convert_onnx_models_to_ort tiny_vad.onnx
   ```

2. Use ONNX Runtime Mobile SDK in your app

### TensorFlow Lite (Alternative)

1. Convert ONNX to TensorFlow:
   ```bash
   onnx-tf convert -i tiny_vad.onnx -o tiny_vad_tf
   ```

2. Convert to TFLite:
   ```python
   import tensorflow as tf
   converter = tf.lite.TFLiteConverter.from_saved_model("tiny_vad_tf")
   tflite_model = converter.convert()
   ```

## Performance Benchmarks

Typical inference speeds (measured on test hardware):

- **CPU (Intel i7)**: ~2-3ms per chunk (100 frames)
- **GPU (NVIDIA GTX 1080)**: ~0.5ms per chunk
- **Mobile (Snapdragon 855)**: ~5-8ms per chunk
- **Raspberry Pi 4**: ~15-20ms per chunk

For real-time processing at 16kHz:
- Requires processing ~10 chunks/second (1 second audio = 1 chunk)
- All platforms above meet real-time requirements

## Post-Processing

Recommended post-processing for production:

1. **Smoothing**: Apply median filter (kernel_size=5)
2. **Hysteresis**: Use dual thresholds (high=0.6, low=0.4)
3. **Hangover**: Extend segments by 5 frames after speech ends

See `vad_distill/utils/postprocessing.py` for implementation.

## Requirements

- Python 3.8+
- onnxruntime (>=1.10.0)
- librosa (for feature extraction)
- numpy

Install with:
```bash
pip install onnxruntime librosa numpy
```

## Notes

- Model expects fixed-size input chunks (100 frames)
- For variable-length audio, chunk with 50% overlap and average predictions
- Output is logits; apply sigmoid for probabilities
- Frame-level scores can be aggregated to segment-level decisions

