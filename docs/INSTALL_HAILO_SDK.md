# Installing Hailo SDK for HEF Compilation

**IMPORTANT:** HEF files are built in **WSL2 Ubuntu 22.04**, not Windows directly.

## WSL2 Installation

### 1. Install WSL2 Ubuntu 22.04

If not already installed:

```bash
wsl --install -d Ubuntu-22.04
```

### 2. Download Hailo SDK

1. Go to: https://hailo.ai/developer-zone/model-zoo/
2. Sign up / log in to Hailo Developer Portal
3. Download **Hailo Dataflow Compiler** for Linux (Ubuntu 22.04)

### 3. Install in WSL2

1. Extract the downloaded SDK in WSL2
2. Follow Hailo's Linux installation instructions
3. Create virtual environment (recommended):
   ```bash
   python3 -m venv ~/hailo310
   source ~/hailo310/bin/activate
   ```

### 4. Verify Installation

```bash
# In WSL2
source ~/hailo310/bin/activate
hailo --version
```

Expected: Hailo Dataflow Compiler version (e.g., 3.30.0)

### 5. Compile ONNX → HEF

See `BUILD_HEF.md` in project root for the complete 3-step compilation process.

## Windows Native (Not Recommended)

**Note:** Windows native installation is not used in this project. All HEF compilation happens in WSL2.

If you need Windows native installation for other purposes:

```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  hailoai/hailo-model-zoo:latest \
  hailomz compile \
    --model-path /workspace/deployment/models/type_classifier_480x170_single.onnx \
    --hw-arch hailo8 \
    --output /workspace/hef/type_classifier_480x170.hef
```

## Notes

- Hailo SDK requires manual installation (not available via pip)
- Windows support may vary - check Hailo documentation for latest Windows instructions
- Linux/WSL might be easier if Windows installation fails
- Docker is a reliable alternative if native installation doesn't work

## Troubleshooting

If installation fails:

1. Check Hailo's system requirements
2. Try WSL (Windows Subsystem for Linux)
3. Use Docker container
4. Use a Linux machine for compilation
