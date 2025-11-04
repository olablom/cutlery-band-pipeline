# Dataset Documentation

## Dataset v1 (Current)

**Preprocessing rules:**
- **Fall 1 - Studio/rig images** (exakt 1440×1080): Crop Y=[480:992] (512 px from lower part)
- **Fall 2 - High-res images** (min(h,w) ≥ 2000, iPhone/DSLR): Center-crop 512 px vertically
- **Fall 3 - Other images**: Take top 512 px
- All images resized to: 480×170

**Source images:**
- Total: 1586 images
- Studio/rig: ~1500 images (1440×1080) in `fork/`, `knife/`, `spoon/` subdirectories
- iPhone/other: 86 images in root of `dataset/raw/`

**Processed output:**
- All images: 480×170 (width × height)
- Location: `dataset/processed/`
- Structure preserves original directory layout

## Crop Parameters

### Studio/Rig Images (1440×1080)
```python
CROP_Y0 = 480  # Start from 480px from top
CROP_H = 512   # Take 512px height
# Result: Y=[480:992] from 1080 height
```

### Other Landscape Images
- Take top 512 px (or full height if < 512)
- Preserves full width

### Portrait/iPhone Images
- Center-crop 512 px vertically
- Preserves full width

## Notes

- Original images are never modified (stored in `dataset/raw/`)
- All preprocessing is deterministic and reproducible
- Can regenerate `dataset/processed/` by running `scripts/preprocess_dataset.py`

