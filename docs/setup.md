# Setup Guide

## PC Setup (Windows)

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Git Bash/CMD):**
```bash
.venv/Scripts/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import cv2; import numpy; import onnxruntime; print('✓ All dependencies installed')"
```

### 5. Test Inference

```bash
python deployment/scripts/infer_fast.py dataset/processed/fork/2025-10-30_14-53-05_rot0__batch1/fork_rot0__b01_0001.jpg
```

## Optional: Development Dependencies

If you need to export ONNX models from PyTorch checkpoints:

```bash
pip install -r requirements-dev.txt
```

## Raspberry Pi Setup

See `docs/pi_setup.md` for Pi-specific instructions.

## Troubleshooting

**Import errors:**
- Ensure venv is activated
- Reinstall: `pip install -r requirements.txt --force-reinstall`

**Permission errors (Windows):**
- Run PowerShell as Administrator
- Or use: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

