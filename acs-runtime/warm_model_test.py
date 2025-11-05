#!/usr/bin/env python3
# warm_model_test.py

from pathlib import Path
import time
import json

from PIL import Image
import onnxruntime as ort
import numpy as np

MODEL_PATH = "models/type_classifier.onnx"
LABELS_PATH = "models/type_labels.json"

# 1) ladda modell + labels en gång
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
labels = json.loads(Path(LABELS_PATH).read_text())

# 2) plocka 200 bilder från ditt processed-träd
# PC path (adjust for Pi: /home/pi/cutlery-band-pipeline/dataset/processed/...)
roots = [
    Path("../dataset/processed/fork"),
    Path("../dataset/processed/knife"),
    Path("../dataset/processed/spoon"),
]

imgs = []
for r in roots:
    if r.exists():
        imgs.extend(list(r.rglob("*.jpg")))

imgs = imgs[:200]  # ta 200 första

if not imgs:
    print("Error: No images found. Check paths.")
    exit(1)

print(f"Found {len(imgs)} images")


def to_np(img: Image.Image):
    img = img.resize((480, 170))
    arr = np.array(img).astype("float32") / 255.0
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    arr = (arr - mean) / std
    return arr


input_name = session.get_inputs()[0].name

t0 = time.time()
latencies = []

for i, p in enumerate(imgs):
    im = Image.open(p).convert("RGB")
    arr = to_np(im)
    t1 = time.time()
    _ = session.run(None, {input_name: arr})
    lat_ms = (time.time() - t1) * 1000.0
    latencies.append(lat_ms)
    
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(imgs)} images...")

total_s = time.time() - t0

latencies.sort()
n = len(latencies)


def pct(q):
    idx = int(q * n)
    idx = min(idx, n - 1)
    return latencies[idx]


print(f"\nran {n} images in {total_s:.2f}s ({total_s/n*1000:.2f} ms/img avg incl. python loop)")
print(f"pure inference ms: min={latencies[0]:.2f} p50={pct(0.5):.2f} p90={pct(0.9):.2f} max={latencies[-1]:.2f}")

# Compare with baseline
baseline_avg = 54.56  # From previous analysis
warm_avg = sum(latencies) / len(latencies)
improvement = ((baseline_avg - warm_avg) / baseline_avg) * 100
print(f"\nComparison:")
print(f"  Baseline (cold, 1 process per image): ~54.56 ms")
print(f"  Warm model (same process): {warm_avg:.2f} ms")
print(f"  Improvement: {improvement:.1f}%")
print(f"  Overhead saved: ~{baseline_avg - warm_avg:.2f} ms per image")
