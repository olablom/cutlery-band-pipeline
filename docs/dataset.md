# Dataset Documentation

## Dataset v1 (Current)

**Preprocessing rules:**
- Only rig/studio images from `fork/`, `knife/`, `spoon/` subdirectories
- All are 1440×1080 → Crop Y=[160:672] (512 px, starting 160px from top)
- Resized to: 480×170
- iPhone/other images ignored (moved to `dataset/raw/extra_phone/`)

**Crop justification:**
- Y0=160 centers cutlery in the processed 480×170 image
- Cutlery appears in the belt region (middle of image), not at top or bottom
- Can be fine-tuned to 170-180 if more "floor" is desired, but 160 is production-ready

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
RIG_Y0 = 160   # Start from 160px from top (centers cutlery in output)
RIG_H = 512    # Take 512px height
# Result: Y=[160:672] from 1080 height
# Fine-tuning: 170-180 for more floor, but 160 is production-ready
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

