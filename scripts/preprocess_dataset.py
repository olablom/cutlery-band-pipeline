#!/usr/bin/env python3
# scripts/preprocess_dataset.py

"""
Preprocess all images in dataset/raw/ (recursively) and write 480x170 images
to dataset/processed/.

Rules:
1. Rigg/studio: 1440x1080  -> crop y=480..992
2. High-res (iPhone/DSLR): min(h, w) >= 2000 -> center-crop 512 vertically
3. Other: take top 512 px

All are then resized to 480x170.
"""

from pathlib import Path
import cv2

SRC = Path("dataset/raw")
DST = Path("dataset/processed")
DST.mkdir(parents=True, exist_ok=True)

TARGET_W, TARGET_H = 480, 170

def resize_and_save(crop, out_path: Path):
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

    # 1) exakt vår studiokamera
    if w == 1440 and h == 1080:
        CROP_Y0 = 480
        CROP_H = 512
        y1 = min(CROP_Y0 + CROP_H, h)
        crop = img[CROP_Y0:y1, 0:1440]

    # 2) stora mobil/DSLR-bilder -> mitt-crop
    elif min(h, w) >= 2000:
        CROP_H = 512
        if h > CROP_H:
            start_y = (h - CROP_H) // 2
            end_y = start_y + CROP_H
        else:
            start_y, end_y = 0, h
        crop = img[start_y:end_y, 0:w]

    # 3) allt annat -> topp 512
    else:
        CROP_H = min(512, h)
        crop = img[0:CROP_H, 0:w]

    rel = img_path.relative_to(SRC)
    out_path = DST / rel
    resize_and_save(crop, out_path)

print("Preprocessing complete.")
