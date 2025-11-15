# VAD Distillation

A Python package for Voice Activity Detection (VAD) distillation using teacher-student architecture. This project distills knowledge from a pretrained FSMN-VAD teacher model into two lightweight student models for efficient deployment.

## Overview

This project implements a complete VAD distillation pipeline:

1. **Teacher Labeling**: Uses a pretrained FSMN-VAD teacher model (Chinese, 16kHz) to generate frame-level VAD probabilities and segment boundaries from audio corpus
2. **Student A (Tiny VAD)**: Lightweight frame-level VAD model using 1D CNN over log-mel features
3. **Student B (Chunk Trigger)**: Chunk boundary detection model that predicts optimal segmentation points from VAD probability windows
4. **ONNX Export**: Exports both student models to ONNX format for deployment

## Project Structure

```
VAD_Training/
├── teacher/                      # Teacher model files (model.pt, config.yaml, etc.)
├── vad_distill/                 # Python package
│   ├── teacher.py               # Teacher model wrapper
│   ├── distill/                 # Student models
│   │   ├── tiny_vad/            # Student A: Frame-level VAD
│   │   └── chunk_trigger/       # Student B: Chunk trigger
│   ├── utils/                   # Audio I/O, features, labeling utilities
│   ├── configs/                 # YAML configuration files
│   └── scripts/                 # CLI entry points
├── models/                      # Training outputs (checkpoints, ONNX)
│   ├── tiny_vad/
│   └── chunk_trigger/
└── data/                        # Data directories (manifests, labels)
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

Key dependencies: `torch`, `funasr`, `modelscope`, `librosa`, `soundfile`, `numpy`, `scipy`, `pyyaml`, `tqdm`

## Quick Start

### 1. Prepare Data

Create a manifest file (JSONL or CSV) listing audio files:

**JSONL format**:
```json
{"utt_id": "utt001", "wav_path": "path/to/audio1.wav"}
{"utt_id": "utt002", "wav_path": "path/to/audio2.wav"}
```

**CSV format**:
```csv
utt_id,wav_path
utt001,path/to/audio1.wav
utt002,path/to/audio2.wav
```

### 2. Configure Paths

Edit configuration files in `vad_distill/configs/`:
- `dataset.yaml`: Set manifest path and dataset root
- `teacher_fsmn.yaml`: Set teacher model directory path (default: `teacher/` in project root)
- `student_tiny_vad.yaml`: Adjust training hyperparameters
- `student_chunk.yaml`: Adjust training hyperparameters

### 3. Run Teacher Labeling

Generate frame-level probabilities and segment boundaries:

```bash
python -m vad_distill.scripts.run_teacher_labeling
```

Outputs:
- `data/teacher_labels/frame_probs/{utt_id}.npy` - Frame-level VAD probabilities
- `data/teacher_labels/segments/{utt_id}.json` - Segment boundaries

### 4. Train Student Models

**Train Student A (Tiny VAD):**
```bash
python -m vad_distill.scripts.train_tiny_vad
```

**Train Student B (Chunk Trigger):**
```bash
python -m vad_distill.scripts.train_chunk_model
```

Checkpoints saved to `models/tiny_vad/checkpoints/` and `models/chunk_trigger/checkpoints/`

### 5. Export to ONNX

Export both student models:

```bash
python -m vad_distill.scripts.export_all_onnx
```

ONNX models saved to `models/tiny_vad/onnx/` and `models/chunk_trigger/onnx/`

## Configuration

### Teacher Model (`configs/teacher_fsmn.yaml`)

```yaml
teacher:
  model_dir: "teacher"  # Path to teacher model directory in project root
  device: "cpu"  # or "cuda"
  frame_hop: 0.01
  frame_len: 0.025
```

### Student A (`configs/student_tiny_vad.yaml`)

Key parameters:
- `model.n_mels`: Number of mel filter banks (default: 80)
- `model.hidden_dims`: CNN channel dimensions
- `training.batch_size`, `learning_rate`, `num_epochs`

### Student B (`configs/student_chunk.yaml`)

Key parameters:
- `model.window_size`: Number of frames in input window
- `model.model_type`: "mlp" or "cnn"
- `data.positive_margin`: Time margin for positive labels (seconds)

## Usage Examples

### Using the Teacher Model

```python
from vad_distill.teacher import TeacherFSMNVAD
from vad_distill.utils.audio_io import load_wav

teacher = TeacherFSMNVAD(model_dir="teacher", device="cpu")
wav = load_wav("path/to/audio.wav", target_sr=16000)
frame_prob, segments = teacher.infer(wav, sr=16000)
```

### Training Student A

```python
from vad_distill.utils.config import load_yaml
from vad_distill.distill.tiny_vad.train import train_tiny_vad

config = load_yaml("vad_distill/configs/student_tiny_vad.yaml")
train_tiny_vad(config)
```

## Model Architecture

### Student A: Tiny VAD

- **Input**: Log-mel spectrogram (time, 80 mel bins)
- **Architecture**: 1D CNN with multiple convolutional layers
- **Output**: Frame-level speech probability logits (time, 1)
- **Loss**: BCEWithLogitsLoss

### Student B: Chunk Trigger

- **Input**: Window of VAD probabilities (window_size, 1)
- **Architecture**: MLP or 1D CNN
- **Output**: Scalar logit for chunk boundary trigger
- **Loss**: BCEWithLogitsLoss

## Data Flow

1. Raw Audio → Teacher Model → Frame Probabilities + Segments
2. Frame Probabilities + Segments → Student A Training → Tiny VAD Model
3. Frame Probabilities + Segments → Student B Training → Chunk Trigger Model
4. Trained Models → ONNX Export → Deployment Models

## Requirements

- Teacher model directory (`teacher/` in project root) must contain:
  - `model.pt` - Model weights
  - `config.yaml` - Model configuration
  - `am.mvn` - Feature normalization
  - `configuration.json` - Additional configuration

- Audio files should be 16kHz mono WAV files (automatic resampling supported)

- Teacher model inference output format may need adjustment based on FunASR version (see TODO in `vad_distill/teacher.py`)
