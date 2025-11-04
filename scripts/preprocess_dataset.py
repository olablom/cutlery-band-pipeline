#!/usr/bin/env python3
# scripts/preprocess_dataset.py

"""
Preprocess only the rig/studio dataset:
- dataset/raw/fork/**.jpg
- dataset/raw/knife/**.jpg
- dataset/raw/spoon/**.jpg

All other images (iPhone/misc) are ignored.

Output: same relative structure under dataset/processed/, resized to 480x170.
"""

from pathlib import Path
import cv2

SRC = Path("dataset/raw")
DST = Path("dataset/processed")
DST.mkdir(parents=True, exist_ok=True)

TARGET_W, TARGET_H = 480, 170

RIG_FOLDERS = {"fork", "knife", "spoon"}

for img_path in SRC.rglob("*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    # check top-level class folder
    try:
        top = img_path.relative_to(SRC).parts[0]
    except ValueError:
        continue

    if top not in RIG_FOLDERS:
        # skip iPhone/misc at root
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    # we know these are the 1440x1080 rig images
    # crop lower part
    CROP_Y0 = 480
    CROP_H = 512
    y1 = min(CROP_Y0 + CROP_H, h)
    crop = img[CROP_Y0:y1, 0:1440]

    # resize to model input
    resized = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    rel = img_path.relative_to(SRC)
    out_path = DST / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), resized)

print("Preprocessing complete (rig images only).")
