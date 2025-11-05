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

## Next Steps

After verifying on Pi:
1. Add manufacturer/registry matching
2. Integrate C++ layer for STM32 communication

