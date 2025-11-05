# Pi Transfer Guide

Complete guide for transferring and deploying ACS runtime to Raspberry Pi.

## Prerequisites

### On PC (before transfer)

- [ ] HEF compiled in WSL2 (see `BUILD_HEF.md` for 3-step process)
- [ ] HEF file exists in `hef/type_classifier_480x170.hef`
- [ ] All code committed and tested

### On Pi (before transfer)

- [ ] Python 3.8+ installed
- [ ] HailoRT installed (for Hailo backend)
- [ ] Network connectivity to Pi

## Step 1: Create Transfer Package

### Option A: Zip Package (Recommended)

```bash
# On PC
cd acs-runtime
zip -r ../acs-runtime-pi.zip . \
  -x "*.pyc" \
  -x "__pycache__/*" \
  -x "logs/*" \
  -x "*.log" \
  -x "inference_log.jsonl"

# Include HEF if available (built in WSL2, see BUILD_HEF.md)
cd ..
zip acs-runtime-pi.zip hef/type_classifier_480x170.hef 2>/dev/null || echo "HEF not found - build in WSL2 using BUILD_HEF.md"
```

### Option B: rsync (Alternative)

```bash
# On PC
rsync -avz --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'logs' \
  --exclude 'inference_log.jsonl' \
  acs-runtime/ pi@<pi-ip>:~/cutlery-band-pipeline/acs-runtime/
```

## Step 2: Transfer to Pi

### Using WeTransfer / Manual Transfer

1. Upload `acs-runtime-pi.zip` to WeTransfer or similar
2. On Pi: Download and extract:
   ```bash
   cd ~/Downloads
   unzip acs-runtime-pi.zip -d ~/cutlery-band-pipeline/
   ```

### Using SCP

```bash
# On PC
scp acs-runtime-pi.zip pi@<pi-ip>:~/Downloads/

# On Pi
cd ~/Downloads
unzip acs-runtime-pi.zip -d ~/cutlery-band-pipeline/
```

### Using rsync

```bash
# On PC (already shown above)
rsync -avz acs-runtime/ pi@<pi-ip>:~/cutlery-band-pipeline/acs-runtime/
```

## Step 3: Transfer HEF File (if available)

```bash
# On PC
scp hef/type_classifier_480x170.hef pi@<pi-ip>:~/cutlery-band-pipeline/acs-runtime/models/type_classifier.hef

# Or via zip
cd hef
zip type_classifier_480x170.hef.zip type_classifier_480x170.hef
# Transfer zip, then on Pi:
cd ~/Downloads
unzip type_classifier_480x170.hef.zip
mv type_classifier_480x170.hef ~/cutlery-band-pipeline/acs-runtime/models/type_classifier.hef
```

## Step 4: Setup on Pi

### Install Dependencies

```bash
cd ~/cutlery-band-pipeline/acs-runtime

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python packages
pip install onnxruntime pillow numpy pyyaml opencv-python

# Install HailoRT (if using Hailo backend)
# Follow Hailo SDK installation guide for Pi
```

### Configure Runtime

#### For ONNX Backend (Default)

```bash
# Edit runtime_config.yaml
nano runtime_config.yaml
```

Verify:
```yaml
inference_backend: "onnx"
model_path: "models/type_classifier.onnx"
```

#### For Hailo Backend

```bash
# Edit runtime_config.yaml
nano runtime_config.yaml
```

Update:
```yaml
inference_backend: "hailo"
hef_path: "models/type_classifier.hef"
```

Ensure HEF file exists:
```bash
ls -lh models/type_classifier.hef
```

## Step 5: Verify Installation

### Test ONNX Backend

```bash
cd ~/cutlery-band-pipeline/acs-runtime
source .venv/bin/activate

# Test with sample image
python3 main.py ../dataset/processed/fork/2025-10-30_14-53-05_rot0__batch1/fork_rot0__b01_0001.jpg

# Expected output:
# [main] Using backend: onnx
# [inference] X.XX ms
# 41435349... (hex frame)
```

### Test Hailo Backend (if available)

```bash
# Update config first (see above)
python3 main.py ../dataset/processed/fork/2025-10-30_14-53-05_rot0__batch1/fork_rot0__b01_0001.jpg

# Expected output:
# [main] Using backend: hailo
# [inference] X.XX ms (Hailo)
# 41435349... (hex frame)
```

## Step 6: Run Performance Tests

### Warm Model Test

```bash
python3 warm_model_test.py ../dataset/processed/fork
```

Compare:
- ONNX: ~20-50 ms average
- Hailo: ~1-5 ms average

### Full Dataset Test

```bash
# Run full test
find ../dataset/processed -name "*.jpg" | while read img; do
  python3 main.py "$img" >> /dev/null
done

# Analyze results
python3 analyze_inference_log.py
```

## Troubleshooting

### Import Errors

```bash
# Verify all dependencies installed
pip list | grep -E "(onnxruntime|pillow|numpy|pyyaml|opencv)"

# Reinstall if needed
pip install --force-reinstall onnxruntime pillow numpy pyyaml opencv-python
```

### Hailo Backend Not Working

1. Check HailoRT installed:
   ```bash
   python3 -c "from hailo_platform import Device, HEF; print('OK')"
   ```

2. Check device available:
   ```bash
   ls /dev/hailo*
   ```

3. Verify HEF file:
   ```bash
   ls -lh models/type_classifier.hef
   ```

### Model Not Found

```bash
# Verify model paths
ls -lh models/type_classifier.onnx
ls -lh models/type_labels.json

# Check config
cat runtime_config.yaml
```

### Permission Issues

```bash
# Make scripts executable
chmod +x *.py

# Ensure write permissions for logs
mkdir -p logs
chmod 755 logs
```

## File Structure on Pi

After transfer, Pi should have:

```
~/cutlery-band-pipeline/acs-runtime/
├── main.py
├── classifier.py
├── classifier_hailo.py
├── decision_engine.py
├── plc_packet.py
├── capture.py
├── utils.py
├── runtime_config.yaml
├── INPUT_SPEC.md
├── README.md
├── config/
│   ├── thresholds.yaml
│   └── plc_actions.yaml
├── models/
│   ├── type_classifier.onnx
│   ├── type_classifier.hef  # If using Hailo
│   └── type_labels.json
├── registry/
│   ├── fork.json
│   ├── knife.json
│   └── spoon.json
└── logs/  # Created automatically
```

## Next Steps After Deployment

1. Run baseline tests with ONNX backend
2. Compare performance with Hailo backend (if available)
3. Set up logging for production monitoring
4. Configure registry with real manufacturer data
5. Integrate with C++ layer for STM32 communication

