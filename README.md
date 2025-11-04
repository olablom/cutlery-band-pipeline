# Cutlery Band Pipeline

High-performance image processing pipeline for cutlery classification on Raspberry Pi 5 + Hailo/Coral.

## Structure

- `dataset/raw/` - Original images (never modify)
- `dataset/processed/` - Preprocessed images (cropped/resized)
- `deployment/models/` - ONNX models for inference
- `deployment/labels/` - Label mappings
- `deployment/scripts/` - Deployment scripts
- `scripts/` - Preprocessing and training scripts
- `reports/` - Training reports and metrics
- `docs/` - Documentation
- `test_images/` - Test images for validation

