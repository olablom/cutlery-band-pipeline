# Dataset Documentation

## Dataset v1 (Current)

**Preprocessing rules:**
- Only rig/studio images from `fork/`, `knife/`, `spoon/` subdirectories
- All are 1440×1080 → Crop Y=[480:992] (512 px from lower part)
- Resized to: 480×170
- iPhone/other images ignored (moved to `dataset/raw/extra_phone/`)

**Source images:**
- Total processed: 1500 images
- All from rig/studio setup (1440×1080)
- Distributed: `fork/`, `knife/`, `spoon/` subdirectories

**Processed output:**
- All images: 480×170 (width × height)
- Location: `dataset/processed/`
- Structure preserves original directory layout (fork/, knife/, spoon/)

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

