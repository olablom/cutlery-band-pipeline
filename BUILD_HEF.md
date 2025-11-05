# Building HEF Files for Hailo-8

**IMPORTANT:** This is the **only** correct way to build HEF files in this project. Do not use Docker, `hailomz`, or other methods.

## Overview

- **HEF files are built in WSL2 Ubuntu 22.04** with Hailo Dataflow Compiler 3.30.0
- **Process:** 3-step pipeline: (1) parser → (2) optimize → (3) compiler
- **Always run from:** `/mnt/c/Users/olabl/Documents/GitHub/cutlery-band-pipeline`
- **Output:** HEF files go in `./hef/`

## Prerequisites

1. WSL2 Ubuntu 22.04 installed
2. Hailo Dataflow Compiler 3.30.0 installed in WSL
3. Virtual environment activated: `source ~/hailo310/bin/activate`

## Step-by-Step Process

### Step 0: Fix ONNX (if needed)

If your ONNX model is missing `kernel_shape`, fix it first:

```bash
# In WSL
cd /mnt/c/Users/olabl/Documents/GitHub/cutlery-band-pipeline
source ~/hailo310/bin/activate

python3 fix_kernel_shape.py \
  --in deployment/models/YOUR_MODEL.onnx \
  --out deployment/models/YOUR_MODEL_fixed.onnx
```

### Step 1: Parse ONNX → HAR

```bash
hailo parser onnx \
  deployment/models/YOUR_MODEL_fixed.onnx \
  --hw-arch hailo8
```

This creates: `YOUR_MODEL_fixed.har` in repo root

### Step 2: Optimize HAR

```bash
hailo optimize \
  --hw-arch hailo8 \
  --use-random-calib-set \
  --output-har-path hef/YOUR_MODEL_optimized.har \
  YOUR_MODEL_fixed.har
```

This creates: `hef/YOUR_MODEL_optimized.har`

### Step 3: Compile HAR → HEF

```bash
hailo compiler \
  --hw-arch hailo8 \
  --output-dir hef \
  hef/YOUR_MODEL_optimized.har
```

This creates: `hef/YOUR_MODEL.hef` (or similar name)

### Step 4: Rename (optional)

If you want a standard name:

```bash
cp hef/YOUR_MODEL.hef hef/cutlery_hailo.hef
```

## Example: Compiling Type Classifier

```bash
# 0) Setup
source ~/hailo310/bin/activate
cd /mnt/c/Users/olabl/Documents/GitHub/cutlery-band-pipeline

# 1) Fix ONNX (if needed)
python3 fix_kernel_shape.py \
  --in deployment/models/type_classifier_480x170_single.onnx \
  --out deployment/models/type_classifier_480x170_single_fixed.onnx

# 2) Parse
hailo parser onnx \
  deployment/models/type_classifier_480x170_single_fixed.onnx \
  --hw-arch hailo8

# 3) Optimize
hailo optimize \
  --hw-arch hailo8 \
  --use-random-calib-set \
  --output-har-path hef/type_classifier_optimized.har \
  type_classifier_480x170_single_fixed.har

# 4) Compile
hailo compiler \
  --hw-arch hailo8 \
  --output-dir hef \
  hef/type_classifier_optimized.har

# 5) Rename to standard name
cp hef/type_classifier_*.hef hef/type_classifier_480x170.hef
```

## Compiling Multiple Models

For **type classifier** and **manufacturer classifier**:

1. Run the same 4 steps for each ONNX file
2. You'll get:
   - `hef/type_classifier_480x170.hef`
   - `hef/manufacturer_classifier_480x170.hef`

## Deployment to Pi

After building HEF on PC/WSL:

```bash
# From WSL or Windows
scp hef/type_classifier_480x170.hef pi@raspberrypiwd2:~/cutlery-band-pipeline/acs-runtime/models/type_classifier.hef
```

On Pi, the runtime will automatically use the HEF file when `inference_backend: "hailo"` is set in `runtime_config.yaml`.

## Important Notes

- **PC/WSL builds** - Pi only runs
- **Always use the 3-step process** - parser → optimize → compiler
- **Do NOT use Docker** for HEF compilation in this project
- **Do NOT use `hailomz`** - use `hailo` CLI commands directly
- If HEF already exists in `./hef/`, you can use it directly (assumes it's named `cutlery_hailo.hef`)

## Troubleshooting

### "kernel_shape missing" error
→ Run `fix_kernel_shape.py` first (Step 0)

### "command not found: hailo"
→ Ensure Hailo SDK is installed in WSL and venv is activated

### HEF file not found after compilation
→ Check `hef/` directory - Hailo may have named it differently

