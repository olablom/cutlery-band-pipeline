# Cutlery Band Pipeline

High-performance image processing pipeline for cutlery classification on Raspberry Pi 5 + Hailo/Coral.

## Quick Start

### PC Setup

1. **Setup environment:**
   ```bash
   python -m venv .venv
   .venv/Scripts/activate  # Windows Git Bash/CMD
   # or: .venv\Scripts\Activate.ps1  # PowerShell
   pip install -r requirements.txt
   ```

2. **Preprocess images:**
   ```bash
   python scripts/preprocess_dataset.py
   ```

3. **Test inference:**
   ```bash
   python deployment/scripts/infer_fast.py dataset/processed/<bild>.jpg
   ```

4. **Run benchmark:**
   ```bash
   python scripts/benchmark_onnx.py deployment/models/type_classifier_480x170.onnx dataset/processed/<bild>.jpg 200
   ```

See `docs/setup.md` for detailed setup instructions.

### Pi Setup
1. Sync to Pi: `./scripts/sync_to_pi.sh <pi-ip>` or manually via `scp`
2. Run inference: `python3 deployment/scripts/infer_fast.py dataset/processed/<bild>.jpg`
3. Run benchmark: `python3 scripts/benchmark_onnx.py deployment/models/type_classifier_480x170.onnx dataset/processed/<bild>.jpg 200`

See `docs/pi_setup.md` for detailed Pi setup instructions.

## Structure

- `dataset/raw/` - Original images (never modify)
- `dataset/processed/` - Preprocessed images (cropped/resized to 480x170)
- `deployment/models/` - ONNX models for inference
- `deployment/labels/` - Label mappings
- `deployment/scripts/` - Deployment scripts
- `scripts/` - Preprocessing, training, and export scripts
- `reports/` - Training reports and metrics
- `docs/` - Documentation
- `test_images/` - Test images for validation

## Image Processing Pipeline

**Source formats:**
- Studio/rig images: 1440×1080 → crop Y=[160:672] (512 px, centers cutlery)
- Only rig images from `fork/`, `knife/`, `spoon/` are processed
- iPhone/other images excluded (moved to `dataset/raw/extra_phone/`)

**Preprocessing:**
- All images resized to: 480×170 (width × height)
- Homogeneous dataset: 1500 rig images ready for training

**Model input:**
- ONNX expects: (batch, 3, 170, 480) → CHW format
- Inference script handles resize and transpose automatically

## Current Status

- ✅ Dataset: 1500 rig images preprocessed to 480x170 (fork/knife/spoon only)
- ✅ Training scripts: Ready in `src/` (requires PyTorch with CUDA for RTX 5090)
- ✅ Model export: Script ready in `scripts/export_trained_onnx.py`
- ✅ PC Benchmark: Mean 5.0ms, P95 8.0ms (old model)
- ⏳ Training: Pending (install CUDA PyTorch first)
- ⏳ Pi Benchmark: Pending (see `docs/pi_setup.md`)

## Next Steps

1. **Install PyTorch with CUDA** (for RTX 5090):
   ```bash
   .venv/Scripts/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Train model:**
   ```bash
   python src/train_480x170.py
   ```

3. **Export ONNX:**
   ```bash
   python scripts/export_trained_onnx.py
   ```

See `docs/training_setup.md` for detailed instructions.

