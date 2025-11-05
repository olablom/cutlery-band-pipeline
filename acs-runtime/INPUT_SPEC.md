# Input Specification

## Model Input Format

All models (ONNX and Hailo) expect the following input format:

### Image Format
- **Channels**: RGB (3 channels)
- **Resolution**: 480×170 pixels (width × height)
- **Color space**: RGB (not BGR)
- **Data type**: float32
- **Value range**: [0.0, 1.0] (normalized by dividing by 255.0)
- **Layout**: NCHW (batch, channels, height, width)
- **Batch size**: 1

### Preprocessing Steps
1. Load image (RGB)
2. Resize to 480×170 (maintaining aspect ratio or direct resize as needed)
3. Convert to float32 and normalize: `pixel_value / 255.0`
4. Apply ImageNet normalization:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
   - Formula: `(pixel - mean) / std`
5. Transpose from HWC to CHW: `(H, W, C) → (C, H, W)`
6. Add batch dimension: `(C, H, W) → (1, C, H, W)`

### Example
```python
# Input: PIL Image or numpy array (H, W, 3) in RGB
img_rgb = image  # Already RGB, 1440×1080 or similar

# Resize
img_resized = cv2.resize(img_rgb, (480, 170))

# Normalize to [0, 1]
img_normalized = img_resized.astype(np.float32) / 255.0

# ImageNet normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_normalized = (img_normalized - mean) / std

# Transpose and add batch
img_chw = np.transpose(img_normalized, (2, 0, 1))
img_batch = np.expand_dims(img_chw, axis=0)  # (1, 3, 170, 480)
```

### Hailo DFC Configuration
When compiling ONNX to HEF, ensure DFC configuration matches:
- Input shape: `(1, 3, 170, 480)`
- Input normalization: ImageNet (mean/std)
- Input format: RGB float32 [0, 1]

