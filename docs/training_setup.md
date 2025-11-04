# Training Setup

## Prerequisites

### For RTX 5090 / CUDA Training

**Your system:**
- RTX 5090 detected
- CUDA Driver Version: 13.0
- Current PyTorch: CPU-only (needs CUDA version)

**Install PyTorch with CUDA:**

```bash
# Activate venv first
.venv/Scripts/activate

# For CUDA 12.1 (stable, recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 12.4 nightly (latest features)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
```

**Or use helper script (Windows):**
```bash
.venv/Scripts/activate
scripts/install_pytorch_cuda.bat
```

**Verify CUDA:**
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### For CPU Training (fallback)

CPU-only PyTorch is already in `requirements-dev.txt`:
```bash
pip install -r requirements-dev.txt
```

## Training

1. **Activate venv:**
   ```bash
   .venv/Scripts/activate  # Windows Git Bash/CMD
   ```

2. **Train model:**
   ```bash
   python src/train_480x170.py
   ```

   Expected output:
   - Training progress per epoch
   - Best model saved to `checkpoints/best_resnet18_480x170.pth`
   - Final validation accuracy reported

3. **Export ONNX:**
   ```bash
   python scripts/export_trained_onnx.py
   ```

4. **Test inference:**
   ```bash
   python deployment/scripts/infer_fast.py dataset/processed/fork/...jpg
   ```

5. **Benchmark:**
   ```bash
   python scripts/benchmark_onnx.py deployment/models/type_classifier_480x170.onnx dataset/processed/fork/...jpg 200
   ```

## Training Configuration

Current settings in `src/train_480x170.py`:
- Dataset: 1500 images (500 per class)
- Input size: 480×170
- Batch size: 32
- Epochs: 8
- Learning rate: 1e-3
- Validation split: 15%
- Model: ResNet18 (modified for 170px height)

## Notes

- Training uses 15% validation split (random shuffle)
- Model architecture modified for low height (170px): stride=1 in conv1 and maxpool
- Best checkpoint saved based on validation accuracy

