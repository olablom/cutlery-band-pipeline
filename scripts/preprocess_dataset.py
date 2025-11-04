#!/usr/bin/env python3
# scripts/preprocess_dataset.py

from pathlib import Path
import cv2

SRC = Path("dataset/raw")
DST = Path("dataset/processed")
DST.mkdir(parents=True, exist_ok=True)

TARGET_W, TARGET_H = 480, 170

RIG_FOLDERS = {"fork", "knife", "spoon"}

# detta är den viktiga raden
RIG_Y0 = 160      # flytta ned ~160 px från toppen
RIG_H = 512

for img_path in SRC.rglob("*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    rel = img_path.relative_to(SRC)
    top = rel.parts[0]

    if top not in RIG_FOLDERS:
        # skippa extra_phone etc
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    # säkerhet så vi inte går utanför
    y1 = min(RIG_Y0 + RIG_H, h)
    crop = img[RIG_Y0:y1, 0:1440]

    resized = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    out_path = DST / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), resized)

print("Preprocessing complete.")
