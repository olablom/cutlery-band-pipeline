#!/usr/bin/env python3
# deployment/scripts/infer_fast.py
import sys
import time
import json
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "deployment/models/type_classifier_480x170.onnx"
LABELS_PATH = "deployment/labels/type_labels.json"

with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def run_inference(img_path: str):
    img = cv2.imread(img_path)
    # safety resize – ska matcha preprocess
    img = cv2.resize(img, (480, 170), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    input_name = session.get_inputs()[0].name

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: x})[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_idx = int(np.argmax(outputs))
    pred_label = LABELS[str(pred_idx)]
    return {
        "file": img_path,
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "latency_ms": round(latency_ms, 3),
    }

if __name__ == "__main__":
    img_path = sys.argv[1]
    res = run_inference(img_path)
    print(json.dumps(res, indent=2))

