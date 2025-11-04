#!/bin/bash
# scripts/run_pi_benchmark.sh
# Run inference and benchmark on Pi via SSH
# Usage: ./scripts/run_pi_benchmark.sh <pi-ip-address> <image-path>

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./scripts/run_pi_benchmark.sh <pi-ip-address> <image-path>"
    echo "Example: ./scripts/run_pi_benchmark.sh 192.168.1.100 dataset/processed/fork/.../fork_rot0__b01_0001.jpg"
    exit 1
fi

PI_IP="$1"
PI_USER="pi"
PI_PATH="/home/pi/cutlery-band-pipeline"
IMG_PATH="$2"

echo "Running inference on Pi..."
ssh "${PI_USER}@${PI_IP}" "cd ${PI_PATH} && python3 deployment/scripts/infer_fast.py ${IMG_PATH}"

echo ""
echo "Running benchmark on Pi (200 runs)..."
ssh "${PI_USER}@${PI_IP}" "cd ${PI_PATH} && python3 scripts/benchmark_onnx.py deployment/models/type_classifier_480x170.onnx ${IMG_PATH} 200"

