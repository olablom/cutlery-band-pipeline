#!/usr/bin/env python3
# scripts/preprocess_dataset.py

"""
Read all images from dataset/raw/, crop to belt region and resize to 480x170,
then write to dataset/processed/.
"""
import cv2
from pathlib import Path

SRC = Path("dataset/raw")
DST = Path("dataset/processed")

DST.mkdir(parents=True, exist_ok=True)

CROP_X0, CROP_X1 = 0, 1440
CROP_Y0, CROP_H = 284, 512   # adjusted for 1440x1080 sources
TARGET_W, TARGET_H = 480, 170

for img_path in SRC.rglob("*"):
    if not img_path.is_file() or img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue
    
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    h, w = img.shape[:2]
    x1 = min(CROP_X1, w)
    y1 = min(CROP_Y0 + CROP_H, h)
    
    crop = img[CROP_Y0:y1, CROP_X0:x1]
    resized = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    
    # Create unique output filename preserving relative path structure
    rel_path = img_path.relative_to(SRC)
    out_path = DST / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), resized)

print("Done. Wrote processed images to dataset/processed/")

