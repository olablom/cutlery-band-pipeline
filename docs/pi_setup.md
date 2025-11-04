# Raspberry Pi Setup & Benchmark

## Prerequisites

On Pi, ensure you have:
- Python 3.7+
- `opencv-python`: `pip3 install opencv-python`
- `onnxruntime`: `pip3 install onnxruntime`
- `numpy`: `pip3 install numpy`

## Sync Repository to Pi

From your PC (in repo root):

```bash
# Option 1: Full sync (if Pi directory doesn't exist)
scp -r . pi@<pi-ip>:/home/pi/cutlery-band-pipeline

# Option 2: Selective sync (if directory exists)
scp -r deployment scripts dataset/processed reports docs pi@<pi-ip>:/home/pi/cutlery-band-pipeline

# Or use the helper script:
chmod +x scripts/sync_to_pi.sh
./scripts/sync_to_pi.sh <pi-ip>
```

## Verify Files on Pi

SSH into Pi and verify:

```bash
ssh pi@<pi-ip>
cd ~/cutlery-band-pipeline

# Check files exist
ls -lh deployment/models/type_classifier_480x170.onnx
ls -lh deployment/labels/type_labels.json
ls -lh deployment/scripts/infer_fast.py
ls -lh scripts/benchmark_onnx.py
```

## Run Inference Test

On Pi:

```bash
cd ~/cutlery-band-pipeline

# Test with a fork image
python3 deployment/scripts/infer_fast.py \
  dataset/processed/fork/2025-10-30_14-53-05_rot0__batch1/fork_rot0__b01_0001.jpg

# Expected output: JSON with pred_label and latency_ms
# Verify: Should match PC prediction (FORK, idx 0)
```

## Run Benchmark

On Pi:

```bash
python3 scripts/benchmark_onnx.py \
  deployment/models/type_classifier_480x170.onnx \
  dataset/processed/fork/2025-10-30_14-53-05_rot0__batch1/fork_rot0__b01_0001.jpg \
  200
```

Expected results:
- Mean: ~9-15 ms (Pi 5 CPU)
- P95: ~12-20 ms
- Should complete 200 runs without errors

## Update Benchmark Report

After running on Pi, update `reports/benchmark_results.md` with Pi results.

## Troubleshooting

**Import errors:**
- Ensure all Python packages are installed: `pip3 install opencv-python onnxruntime numpy`

**Shape mismatch:**
- Verify model file is correct: `ls -lh deployment/models/type_classifier_480x170.onnx`
- Should be ~43MB

**Permission errors:**
- Make scripts executable: `chmod +x scripts/*.py deployment/scripts/*.py`

