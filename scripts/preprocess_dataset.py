#!/usr/bin/env python3
# scripts/preprocess_dataset.py

"""
Preprocess all images in dataset/raw/ (recursively) and write 480x170 images
to dataset/processed/.

We handle 3 cases:
1. Studio/rig images: 1440x1080 -> crop lower part (y=480..992)
2. Other landscape images: take top 512px
3. Portrait / phone images: center-crop 512px vertically
"""

from pathlib import Path
import cv2

SRC = Path("dataset/raw")
DST = Path("dataset/processed")
DST.mkdir(parents=True, exist_ok=True)

# target model input
TARGET_W, TARGET_H = 480, 170

def save_resized(crop, out_path: Path):
    # resize to (480,170)
    resized = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), resized)

for img_path in SRC.rglob("*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    # case 1: our rig/studio images
    if w == 1440 and h == 1080:
        # use the tuned crop for these
        CROP_Y0 = 480   # you can push this to 500 if you want even lower
        CROP_H = 512
        y1 = min(CROP_Y0 + CROP_H, h)
        crop = img[CROP_Y0:y1, 0:1440]

    # case 2: landscape but not our 1440x1080
    elif w >= h:
        # take top 512 px (most band-like images have object high up)
        CROP_H = min(512, h)
        crop = img[0:CROP_H, 0:w]

    # case 3: portrait / odd aspect
    else:
        # center-crop 512 vertically
        CROP_H = min(512, h)
        start_y = max(0, (h - CROP_H) // 2)
        end_y = start_y + CROP_H
        crop = img[start_y:end_y, 0:w]

    # build output path with same relative structure
    rel = img_path.relative_to(SRC)
    out_path = DST / rel
    save_resized(crop, out_path)

print("Preprocessing complete.")
