#!/usr/bin/env python3
# scripts/benchmark_onnx.py
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

if len(sys.argv) < 3:
    print("usage: python scripts/benchmark_onnx.py <model.onnx> <image> [runs]")
    sys.exit(1)

model_path = sys.argv[1]
img_path = sys.argv[2]
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 200

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

img = cv2.imread(img_path)
img = cv2.resize(img, (480, 170), interpolation=cv2.INTER_AREA)
x = img.astype(np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))[None, ...]
input_name = session.get_inputs()[0].name

times = []
for _ in range(runs):
    t0 = time.perf_counter()
    session.run(None, {input_name: x})
    times.append((time.perf_counter() - t0) * 1000.0)

times = np.array(times)
print(f"Runs: {runs}")
print(f"Mean: {times.mean():.3f} ms")
print(f"P95 : {np.percentile(times, 95):.3f} ms")

