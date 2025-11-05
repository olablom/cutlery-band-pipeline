# ACS Runtime Package

Complete inference pipeline for cutlery classification with ONNX Runtime.

## Contents

- **Full inference pipeline**: ONNX Runtime, thresholds, decision engine, logging, PLC frame generation
- **Latency measurement**: Built-in and logged
- **Big-endian 32-byte frame**: Verified against C++ implementation
- **Configuration-driven**: Change model, thresholds, policies without code changes

## Installation on Raspberry Pi

```bash
# Extract package
unzip acs-runtime.zip
cd acs-runtime

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install onnxruntime pillow numpy pyyaml opencv-python

# Test run
python3 main.py test_images/fork.jpg
```

## Expected Output

- **Log file**: `logs/inference_log.jsonl` with inference results
- **Stdout**: Hex frame starting with `41435349` (ACSI magic)
- **Latency**: Measured in milliseconds (compare with PC values)

## Configuration

- `runtime_config.yaml`: Main configuration (model paths, log paths)
- `config/thresholds.yaml`: Confidence thresholds per class
- `config/plc_actions.yaml`: PLC action templates

## File Structure

```
acs-runtime/
├── main.py              # Entry point
├── capture.py           # Image loading/preprocessing
├── classifier.py        # ONNX inference
├── decision_engine.py   # Decision logic
├── plc_packet.py        # 32-byte frame generation
├── utils.py             # Utilities
├── runtime_config.yaml  # Main config
├── config/              # YAML configs
├── models/              # ONNX model and labels
└── registry/            # Registry (empty, for future use)
```

## Backend Selection

The runtime supports two inference backends (configured in `runtime_config.yaml`):

### ONNX Backend (Default)
- Uses ONNX Runtime (CPU)
- Works on both PC and Pi
- Latency: ~20-50 ms on Pi

### Hailo Backend
- Uses Hailo-8 accelerator (requires Hailo hardware)
- Latency: ~1-5 ms on Pi
- Requires HEF file compiled from ONNX in WSL2

**Building HEF files:** See `BUILD_HEF.md` in project root for the 3-step WSL compilation process.

## Next Steps

After verifying on Pi:
1. Compile ONNX to HEF for Hailo acceleration
2. Test Hailo backend and compare performance
3. Add manufacturer/registry matching
4. Integrate C++ layer for STM32 communication

