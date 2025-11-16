"""Sanity check script to verify VAD system structure and functionality."""

from __future__ import annotations

import sys
from pathlib import Path
import json
import numpy as np
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_chunk_directory():
    """Check that chunk directory exists and contains files."""
    chunks_dir = project_root / "data" / "chunks"
    if not chunks_dir.exists():
        print("ERROR: data/chunks/ directory does not exist")
        return False
    
    chunk_files = list(chunks_dir.glob("chunk_*.npy"))
    if len(chunk_files) == 0:
        print("WARNING: data/chunks/ exists but contains no chunk files")
        print("   Run preprocessing pipeline first")
        return False
    
    print(f"PASS: Chunk directory exists with {len(chunk_files)} chunk files")
    return True


def check_teacher_labels_directory():
    """Check that teacher_labels directory exists."""
    teacher_labels_dir = project_root / "data" / "teacher_labels"
    if not teacher_labels_dir.exists():
        print("ERROR: data/teacher_labels/ directory does not exist")
        return False
    
    # Check for at least one uid directory
    uid_dirs = [d for d in teacher_labels_dir.iterdir() if d.is_dir()]
    if len(uid_dirs) == 0:
        print("WARNING: data/teacher_labels/ exists but contains no uid directories")
        return False
    
    # Check that at least one has required files
    has_required = False
    for uid_dir in uid_dirs[:5]:  # Check first 5
        if (uid_dir / "frame_probs.npy").exists() and (uid_dir / "fbank.npy").exists():
            has_required = True
            break
    
    if not has_required:
        print("WARNING: No uid directories found with frame_probs.npy and fbank.npy")
        return False
    
    print(f"PASS: Teacher labels directory exists with {len(uid_dirs)} uid directories")
    return True


def check_configs():
    """Check that config files contain required fields."""
    config_path = project_root / "vad_distill" / "configs" / "student_tiny_vad.yaml"
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    required_fields = [
        'chunk_size', 'stride', 'n_mels',
        'chunks_dir', 'teacher_labels_dir',
        'batch_size', 'learning_rate', 'epochs',
        'model'
    ]
    
    missing = []
    for field in required_fields:
        if field not in config:
            missing.append(field)
    
    if missing:
        print(f"ERROR: Config missing required fields: {missing}")
        return False
    
    print("PASS: Config file contains all required fields")
    return True


def check_no_legacy_directories():
    """Check that no legacy directories exist in root."""
    legacy_dirs = [
        'data/manifests',
        'data/precomputed_cache',
        'data/tiny_vad_inputs',
        'training_history.json',
        'train_tiny_vad.ipynb',
    ]
    
    found = []
    for legacy_path in legacy_dirs:
        path = project_root / legacy_path
        if path.exists():
            found.append(legacy_path)
    
    if found:
        print(f"ERROR: Legacy directories/files found in root: {found}")
        return False
    
    print("PASS: No legacy directories found in root")
    return True


def check_no_pycache():
    """Check that no __pycache__ directories exist under vad_distill/."""
    vad_distill_dir = project_root / "vad_distill"
    if not vad_distill_dir.exists():
        print("ERROR: vad_distill/ directory does not exist")
        return False
    
    pycache_dirs = list(vad_distill_dir.rglob("__pycache__"))
    if len(pycache_dirs) > 0:
        print(f"ERROR: Found {len(pycache_dirs)} __pycache__ directories under vad_distill/")
        for pc in pycache_dirs[:5]:
            print(f"   - {pc.relative_to(project_root)}")
        return False
    
    print("PASS: No __pycache__ directories found under vad_distill/")
    return True


def check_dataset_loads():
    """Check that dataset loads at least one chunk successfully."""
    try:
        from vad_distill.distill.tiny_vad.dataset import TinyVADChunkDataset
        
        chunks_dir = project_root / "data" / "chunks"
        index_file = chunks_dir / "index.json"
        
        if not index_file.exists():
            print("WARNING: index.json not found, dataset will scan directory")
            index_file = None
        
        dataset = TinyVADChunkDataset(
            chunks_dir=str(chunks_dir),
            index_file=str(index_file) if index_file else None,
            shuffle=False,
        )
        
        if len(dataset) == 0:
            print("ERROR: Dataset is empty")
            return False
        
        # Try to load first chunk
        features, labels = dataset[0]
        
        if features.shape != (100, 80):
            print(f"ERROR: Invalid features shape: {features.shape}, expected (100, 80)")
            return False
        
        if labels.shape != (100,):
            print(f"ERROR: Invalid labels shape: {labels.shape}, expected (100,)")
            return False
        
        print(f"PASS: Dataset loads successfully: {len(dataset)} chunks, first chunk shape OK")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_forward():
    """Check that model forward works for a dummy input."""
    try:
        from vad_distill.distill.tiny_vad.model import build_tiny_vad_model
        from vad_distill.utils.config import load_yaml
        
        config_path = project_root / "vad_distill" / "configs" / "student_tiny_vad.yaml"
        config = load_yaml(str(config_path))
        
        model = build_tiny_vad_model(config)
        model.eval()
        
        # Create dummy input: (batch=1, time=100, freq=80)
        dummy_input = torch.randn(1, 100, 80)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        if output.shape != (1, 100, 1):
            print(f"ERROR: Invalid output shape: {output.shape}, expected (1, 100, 1)")
            return False
        
        print("PASS: Model forward pass works correctly")
        return True
        
    except Exception as e:
        print(f"ERROR: Model forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_loop_step():
    """Check that training loop runs one step."""
    try:
        from vad_distill.distill.tiny_vad.dataset import TinyVADChunkDataset
        from vad_distill.distill.tiny_vad.model import build_tiny_vad_model
        from vad_distill.utils.training import create_optimizer
        from torch.utils.data import DataLoader
        import torch.nn as nn
        
        # Create minimal dataset (use first few chunks only)
        chunks_dir = project_root / "data" / "chunks"
        dataset = TinyVADChunkDataset(
            chunks_dir=str(chunks_dir),
            index_file=None,
            shuffle=False,
        )
        
        if len(dataset) == 0:
            print("WARNING: Cannot test training loop - dataset is empty")
            return False
        
        # Create small dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Build model
        from vad_distill.utils.config import load_yaml
        config_path = project_root / "vad_distill" / "configs" / "student_tiny_vad.yaml"
        config = load_yaml(str(config_path))
        model = build_tiny_vad_model(config)
        
        # Setup optimizer and loss
        optimizer = create_optimizer(model, 'adam', 0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        # Run one step
        model.train()
        features, labels = next(iter(dataloader))
        
        optimizer.zero_grad()
        logits = model(features).squeeze(-1)  # (B, 100)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        print("PASS: Training loop runs one step successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: Training loop step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_evaluation_script():
    """Check that evaluation script can be imported and has required functions."""
    try:
        from vad_distill.scripts.test_single_wav import test_single_wav
        
        # Just check that function exists and is callable
        if not callable(test_single_wav):
            print("ERROR: test_single_wav is not callable")
            return False
        
        print("PASS: Evaluation script imports successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to import evaluation script: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all sanity checks."""
    print("=" * 60)
    print("VAD SYSTEM SANITY CHECK")
    print("=" * 60)
    print()
    
    checks = [
        ("Chunk directory", check_chunk_directory),
        ("Teacher labels directory", check_teacher_labels_directory),
        ("Config files", check_configs),
        ("No legacy directories", check_no_legacy_directories),
        ("No __pycache__ directories", check_no_pycache),
        ("Dataset loads", check_dataset_loads),
        ("Model forward", check_model_forward),
        ("Training loop step", check_training_loop_step),
        ("Evaluation script", check_evaluation_script),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"ERROR: Check '{name}' raised exception: {e}")
            results.append((name, False))
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\nALL CHECKS PASSED - System is ready")
        return 0
    else:
        print(f"\n{total - passed} CHECK(S) FAILED - Please fix issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

