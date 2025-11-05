# Hailo HEF Build Guide

**See `BUILD_HEF.md` in project root for the complete compilation process.**

This document provides context and additional information.

## Quick Reference

The HEF build process uses a **3-step pipeline in WSL2**:

1. **Parser:** ONNX → HAR (intermediate format)
2. **Optimize:** HAR → Optimized HAR (with calibration)
3. **Compiler:** Optimized HAR → HEF (final format)

## Current Models

- **Type Classifier:** `type_classifier_480x170_single.onnx` → `type_classifier_480x170.hef`
- **Manufacturer Classifier:** (when available) → `manufacturer_classifier_480x170.hef`

## Deployment Flow

1. **Build HEF in WSL** (see `BUILD_HEF.md`)
2. **Transfer to Pi:** `scp hef/*.hef pi@raspberrypiwd2:~/cutlery-band-pipeline/acs-runtime/models/`
3. **Update config on Pi:** Set `inference_backend: "hailo"` in `runtime_config.yaml`
4. **Run:** `python3 acs-runtime/main.py <image>`

## Notes

- **PC/WSL builds** - Pi only runs
- HEF files are hardware-specific (Hailo-8)
- Same ONNX model can produce different HEF files for different hardware (`hailo8` vs `hailo8l`)

