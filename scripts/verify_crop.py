#!/usr/bin/env python3
# scripts/verify_crop.py

"""
Quick script to verify crop parameters by comparing original vs processed image dimensions.
Usage: python scripts/verify_crop.py <image_path>
"""
import sys
import cv2
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python scripts/verify_crop.py <image_path>")
    print("Example: python scripts/verify_crop.py dataset/processed/fork/.../fork_rot0__b01_0001.jpg")
    sys.exit(1)

img_path = Path(sys.argv[1])
if not img_path.exists():
    print(f"Error: {img_path} not found")
    sys.exit(1)

# Find corresponding original
if "processed" in str(img_path):
    rel_path = img_path.relative_to("dataset/processed")
    original_path = Path("dataset/raw") / rel_path
    if not original_path.exists():
        print(f"Warning: Original not found at {original_path}")
        print(f"Showing processed image info only:")
        original_path = None
else:
    original_path = img_path

# Load processed image
processed = cv2.imread(str(img_path))
if processed is None:
    print(f"Error: Could not load {img_path}")
    sys.exit(1)

print(f"\nProcessed image: {img_path.name}")
print(f"  Size: {processed.shape[1]}x{processed.shape[0]} (width x height)")
print(f"  Expected: 480x170")

if original_path and original_path.exists():
    original = cv2.imread(str(original_path))
    if original is not None:
        print(f"\nOriginal image: {original_path.name}")
        print(f"  Size: {original.shape[1]}x{original.shape[0]} (width x height)")
        print(f"  Expected: 1440x1080")
        print(f"\nCrop check:")
        # Read current crop params from preprocess script
        import re
        with open('scripts/preprocess_dataset.py', 'r') as f:
            content = f.read()
            match = re.search(r'CROP_Y0.*?(\d+).*?CROP_H.*?(\d+)', content)
            if match:
                y0, h = int(match.group(1)), int(match.group(2))
                print(f"  Current crop: Y=[{y0}:{y0+h}] from height {original.shape[0]}")
            else:
                print(f"  Current crop: Y=[0:512] from height {original.shape[0]}")
        print(f"  If crop is wrong, edit CROP_Y0 and CROP_H in scripts/preprocess_dataset.py")

print(f"\nTo view the image, open it in an image viewer:")
print(f"  {img_path.absolute()}")

