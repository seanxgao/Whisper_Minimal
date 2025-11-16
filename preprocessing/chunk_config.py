"""Chunk configuration constants for VAD training pipeline."""

# Frame extraction parameters
FRAME_LEN = 0.025  # 25ms frame length
FRAME_HOP = 0.01   # 10ms frame hop (100 frames per second)

# Chunk parameters
CHUNK_SIZE = 100   # Number of frames per chunk (1 second at 10ms hop)
CHUNK_STRIDE = 50  # Stride between chunks (0.5 seconds, 50% overlap)

# Feature parameters
N_MELS = 80        # Number of mel filter banks
SAMPLE_RATE = 16000  # Audio sample rate in Hz

# Fixed tensor shapes
FEATURE_SHAPE = (CHUNK_SIZE, N_MELS)  # (100, 80)
LABEL_SHAPE = (CHUNK_SIZE,)            # (100,)

