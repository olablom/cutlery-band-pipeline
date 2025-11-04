# Cutlery Band Pipeline

High-performance image processing pipeline for cutlery classification on Raspberry Pi 5 + Hailo/Coral.

## Quick Start

### PC Setup
1. Preprocess images: `python scripts/preprocess_dataset.py`
2. Test inference: `python deployment/scripts/infer_fast.py dataset/processed/<bild>.jpg`
3. Run benchmark: `python scripts/benchmark_onnx.py deployment/models/type_classifier_480x170.onnx dataset/processed/<bild>.jpg 200`

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

## Current Status

- ✅ Dataset: 1587 images preprocessed to 480x170
- ✅ Model: ResNet18 ONNX exported for 480x170 input
- ✅ PC Benchmark: Mean 5.0ms, P95 8.0ms
- ⏳ Pi Benchmark: Pending (see `docs/pi_setup.md`)

