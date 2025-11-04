#!/bin/bash
# scripts/sync_to_pi.sh
# Sync repository to Raspberry Pi
# Usage: ./scripts/sync_to_pi.sh <pi-ip-address>

if [ -z "$1" ]; then
    echo "Usage: ./scripts/sync_to_pi.sh <pi-ip-address>"
    echo "Example: ./scripts/sync_to_pi.sh 192.168.1.100"
    exit 1
fi

PI_IP="$1"
PI_USER="pi"
PI_PATH="/home/pi/cutlery-band-pipeline"

echo "Syncing to pi@${PI_IP}:${PI_PATH}..."

# Create directory on Pi if it doesn't exist
ssh "${PI_USER}@${PI_IP}" "mkdir -p ${PI_PATH}"

# Sync files (excluding large processed images if needed, but including structure)
scp -r deployment "${PI_USER}@${PI_IP}:${PI_PATH}/"
scp -r scripts "${PI_USER}@${PI_IP}:${PI_PATH}/"
scp -r dataset/processed "${PI_USER}@${PI_IP}:${PI_PATH}/dataset/"
scp -r reports "${PI_USER}@${PI_IP}:${PI_PATH}/" 2>/dev/null || true
scp -r docs "${PI_USER}@${PI_IP}:${PI_PATH}/" 2>/dev/null || true
scp README.md "${PI_USER}@${PI_IP}:${PI_PATH}/" 2>/dev/null || true

echo "✓ Sync complete!"
echo ""
echo "Next steps on Pi:"
echo "  cd ${PI_PATH}"
echo "  python3 deployment/scripts/infer_fast.py dataset/processed/<bild>.jpg"
echo "  python3 scripts/benchmark_onnx.py deployment/models/type_classifier_480x170.onnx dataset/processed/<bild>.jpg 200"

