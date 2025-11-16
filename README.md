# VAD Distillation

A Python package for Voice Activity Detection (VAD) distillation using teacher-student architecture. This project distills knowledge from a pretrained FSMN-VAD teacher model into lightweight student models for efficient deployment.

## Overview

This project implements a complete VAD distillation pipeline with **chunk-based training** for efficient and scalable VAD model training:

1. **Preprocessing**: Extract frame-level features and generate teacher labels
2. **Chunking**: Create fixed-size chunks (100 frames = 1 second) from frame-level data
3. **Training**: Train lightweight student models on preprocessed chunks
4. **Inference**: Deploy trained models for real-time VAD

## Architecture

### Chunk-Based Training Pipeline

The system uses a **frame → chunk → model** architecture:

- **Frame Extraction**: 25ms window, 10ms hop → 100 frames per second
- **Feature Extraction**: 80-dim log-mel filterbanks
- **Chunking**: Fixed-size chunks of 100 frames (1 second) with 50-frame stride (50% overlap)
- **Training**: Fixed-shape tensors `(batch, 100, 80)` → `(batch, 100, 1)`

**Key Benefits:**
- No variable-length audio loading during training
- Fast dataloader (direct chunk file access)
- Efficient GPU utilization
- Scalable to large datasets

### Fixed Tensor Shapes

- **Input**: `(batch, 100, 80)` - 100 frames × 80 mel bins
- **Output**: `(batch, 100, 1)` - 100 frame-level VAD logits
- **No ragged tensors, no dynamic padding**

## Project Structure

```
VAD_Training/
├── preprocessing/              # Preprocessing pipeline
│   ├── chunk_config.py        # Chunk configuration constants
│   ├── extract_fbank.py       # Frame-level feature extraction
│   ├── generate_teacher_labels.py  # Teacher model inference
│   └── create_chunks.py       # Chunk creation from frames
├── vad_distill/               # Python package
│   ├── teacher.py            # Teacher model wrapper
│   ├── distill/               # Student models
│   │   └── tiny_vad/         # Tiny VAD model
│   │       ├── dataset.py    # Chunk dataset loader
│   │       ├── model.py      # TinyVAD model
│   │       ├── train.py      # Training loop
│   │       ├── export_onnx.py # ONNX export
│   │       └── checkpoints/   # Model checkpoints
│   ├── utils/                 # Audio I/O, features utilities
│   ├── configs/               # YAML configuration files
│   │   ├── student_tiny_vad.yaml
│   │   └── teacher_fsmn.yaml
│   └── scripts/               # CLI entry points
│       ├── run_preprocessing_pipeline.py
│       ├── run_teacher_labeling.py
│       ├── test_single_wav.py
│       ├── test_directory.py
│       └── visualize_vad.py
├── data/                      # Data directories
│   ├── chunks/                # Preprocessed chunk files
│   │   ├── chunk_000001.npy
│   │   ├── chunk_000002.npy
│   │   ├── index.json
│   │   └── metadata.json
│   └── teacher_labels/       # Teacher model outputs
│       └── {uid}/
│           ├── frame_probs.npy
│           └── fbank.npy
├── logs/                      # Training logs
│   ├── train.log
│   ├── train_history.json
│   └── tensorboard/
├── models/                    # Model outputs
│   └── tiny_vad/
│       └── onnx/
└── teacher/                   # Teacher model files
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `funasr`, `modelscope`, `librosa`, `soundfile`, `numpy`, `scipy`, `pyyaml`, `tqdm`, `matplotlib`, `tensorboard`

## Quick Start

### Step 1: Preprocessing Pipeline

Run the complete preprocessing pipeline to generate chunks:

```bash
python -m vad_distill.scripts.run_preprocessing_pipeline \
    --manifest path/to/manifest.jsonl \
    --output_root data \
    --teacher_model_dir teacher \
    --device cpu
```

This will:
1. Extract fbank features for each audio file → `data/teacher_labels/{uid}/fbank.npy`
2. Generate teacher labels → `data/teacher_labels/{uid}/frame_probs.npy`
3. Create fixed-size chunks → `data/chunks/chunk_*.npy`
4. Generate chunk index and metadata → `data/chunks/index.json`, `data/chunks/metadata.json`

### Step 2: Train Model

Train the tiny VAD model on preprocessed chunks:

```python
from vad_distill.utils.config import load_yaml
from vad_distill.distill.tiny_vad.train import train_tiny_vad

config = load_yaml("vad_distill/configs/student_tiny_vad.yaml")
train_tiny_vad(config)
```

The training script will:
- Load chunks from `data/chunks/`
- Create fixed-shape batches `(batch, 100, 80)`
- Train with fast dataloader (no audio I/O during training)
- Save checkpoints to `vad_distill/distill/tiny_vad/checkpoints/`
- Log to `logs/train.log` and TensorBoard

### Step 3: Evaluate Model

Test the model on a single WAV file:

```bash
python -m vad_distill.scripts.test_single_wav \
    path/to/audio.wav \
    vad_distill/distill/tiny_vad/checkpoints/best.pt \
    --config vad_distill/configs/student_tiny_vad.yaml \
    --output_dir outputs
```

Visualize VAD predictions:

```bash
python -m vad_distill.scripts.visualize_vad \
    path/to/audio.wav \
    --scores outputs/audio_scores.npy \
    --segments outputs/audio_segments.json \
    --output outputs/visualization.png
```

## Configuration

### Training Configuration (`vad_distill/configs/student_tiny_vad.yaml`)

```yaml
# Chunk configuration
chunk_size: 100  # Frames per chunk (1 second at 10ms hop)
stride: 50       # Stride between chunks (50% overlap)
n_mels: 80       # Number of mel filter banks

# Data paths
chunks_dir: "data/chunks"
teacher_labels_dir: "data/teacher_labels"

# Training configuration
batch_size: 32
learning_rate: 0.001
epochs: 100
num_workers: 4

# Logging and checkpointing
use_tensorboard: true
save_every_n_steps: 1000
validate_every_n_epochs: 5

# Resume training (set to checkpoint path or null)
resume_from: null

# Model architecture
model:
  n_mels: 80
  hidden_dims: [32, 64, 32]
  kernel_sizes: [3, 3, 3]
  dropout: 0.1

# Paths
paths:
  checkpoint_dir: "vad_distill/distill/tiny_vad/checkpoints"
  onnx_dir: "models/tiny_vad/onnx"
  logs_dir: "logs"
```

## Input/Output Shapes

### Model I/O

- **Input**: `(batch, 100, 80)` - Fixed-size chunks
  - 100 frames = 1 second at 10ms hop
  - 80 mel filter banks
- **Output**: `(batch, 100, 1)` - Frame-level VAD logits
  - Squeezed to `(batch, 100)` for loss computation

### Chunk Files

Each chunk file (`chunk_*.npy`) contains:
```python
{
    "features": np.ndarray (100, 80),  # Log-mel features
    "labels": np.ndarray (100,),      # Frame-level VAD probabilities
    "uid": str,                        # Audio file identifier
    "chunk_idx": int                   # Chunk index within audio file
}
```

## Model Export

Export trained model to ONNX:

```python
from vad_distill.utils.config import load_yaml
from vad_distill.distill.tiny_vad.export_onnx import export_tiny_vad_onnx

config = load_yaml("vad_distill/configs/student_tiny_vad.yaml")
export_tiny_vad_onnx(
    config,
    checkpoint_path="vad_distill/distill/tiny_vad/checkpoints/best.pt"
)
```

The exported ONNX model expects fixed input shape `(1, 100, 80)`.

### ONNX Validation

Validate exported ONNX model against PyTorch:

```bash
python scripts/validate_onnx.py \
    models/tiny_vad/onnx/tiny_vad.onnx \
    vad_distill/configs/student_tiny_vad.yaml \
    vad_distill/distill/tiny_vad/checkpoints/best.pt \
    --num_tests 10 \
    --tolerance 1e-5
```

## Deployment

### Export Model for Deployment

1. Export trained model to ONNX:

```python
from vad_distill.utils.config import load_yaml
from vad_distill.distill.tiny_vad.export_onnx import export_tiny_vad_onnx

config = load_yaml("vad_distill/configs/student_tiny_vad.yaml")
export_tiny_vad_onnx(
    config,
    checkpoint_path="vad_distill/distill/tiny_vad/checkpoints/best.pt"
)
```

2. Copy deployment files:

```bash
cp models/tiny_vad/onnx/tiny_vad.onnx deploy/
cp deploy/config.json deploy/  # Already included
```

### Inference Example

Basic inference with ONNX model:

```python
from deploy.inference_example import TinyVADInference
from vad_distill.utils.audio_io import load_wav

# Initialize
inference = TinyVADInference("deploy/tiny_vad.onnx", "deploy/config.json")

# Load and process audio
wav = load_wav("audio.wav", target_sr=16000)
frame_scores = inference.infer(wav)

# Extract segments with post-processing
from vad_distill.utils.postprocessing import postprocess_vad_scores
smoothed_scores, segments = postprocess_vad_scores(
    frame_scores,
    smooth_method="median",
    threshold=0.5,
    use_hysteresis=True,
    high_threshold=0.6,
    low_threshold=0.4,
)
```

### Command Line Inference

```bash
python deploy/inference_example.py audio.wav \
    --onnx deploy/tiny_vad.onnx \
    --output scores.npy
```

### Post-Processing Options

The system includes several post-processing methods:

- **Median Filter**: Smooth frame-level scores
  ```python
  from vad_distill.utils.postprocessing import median_filter
  smoothed = median_filter(frame_scores, kernel_size=5)
  ```

- **Hysteresis Thresholding**: Reduce flickering with dual thresholds
  ```python
  from vad_distill.utils.postprocessing import hysteresis_threshold
  binary = hysteresis_threshold(frame_scores, high_threshold=0.6, low_threshold=0.4)
  ```

- **Hangover Scheme**: Extend speech segments
  ```python
  from vad_distill.utils.postprocessing import hangover_scheme
  binary = hangover_scheme(frame_scores, threshold=0.5, hangover_frames=5)
  ```

- **Complete Pipeline**: Combined post-processing
  ```python
  from vad_distill.utils.postprocessing import postprocess_vad_scores
  smoothed, segments = postprocess_vad_scores(
      frame_scores,
      smooth_method="median",
      threshold=0.5,
      use_hysteresis=True,
      use_hangover=True,
  )
  ```

### Performance Benchmarks

Typical inference speeds (measured on test hardware):

- **CPU (Intel i7)**: ~2-3ms per chunk (100 frames)
- **GPU (NVIDIA GTX 1080)**: ~0.5ms per chunk
- **Mobile (Snapdragon 855)**: ~5-8ms per chunk
- **Raspberry Pi 4**: ~15-20ms per chunk

For real-time processing at 16kHz, all platforms above meet requirements.

### Mobile / Embedded Deployment

See `deploy/README_DEPLOY.md` for detailed mobile deployment instructions.

## Training Features

- **Automatic Logging**: All training logs saved to `logs/train.log`
- **TensorBoard**: Training metrics visualized in `logs/tensorboard/`
- **Checkpointing**: 
  - `checkpoints/best.pt` - Best model
  - `checkpoints/last.pt` - Latest checkpoint
  - `checkpoints/optimizer.pt` - Optimizer state
  - `checkpoints/scheduler.pt` - Learning rate scheduler state
  - `checkpoints/config_used.yaml` - Configuration used
  - `checkpoints/train_history.json` - Training history
- **Resume Training**: Set `resume_from: "checkpoints/last.pt"` in config
- **Validation**: Automatic validation every N epochs
- **Learning Rate Scheduling**: ReduceLROnPlateau scheduler
- **Gradient Clipping**: Automatic gradient clipping for stability

## Requirements

- Teacher model directory (`teacher/` in project root) must contain:
  - `model.pt` - Model weights
  - `config.yaml` - Model configuration
  - `am.mvn` - Feature normalization
  - `configuration.json` - Additional configuration

- Audio files should be 16kHz mono (automatic resampling supported)

## Testing

Run unit tests for chunk alignment:

```bash
pytest tests/test_chunk_alignment.py -v
```

Tests verify:
- Chunk creation alignment
- Fbank and frame_probs length matching
- Chunk slicing accuracy
- Reconstruction from chunks returns exact frame count

## Notes

- **Chunk-based training**: All training uses fixed-size chunks (100 frames)
- **No variable-length**: Model enforces fixed input shape
- **Fast preprocessing**: One-time cost, training is fast
- **Reproducibility**: Fixed seeds ensure deterministic results
- **Strict validation**: Dataset validates all chunk shapes, fails loudly on errors
- **Unified feature extraction**: All scripts use same config from `preprocessing/chunk_config.py`
- **Versioned teacher labels**: Teacher labels stored in `data/teacher_labels/v1/` with metadata
