#!/usr/bin/env python3
# scripts/export_onnx.py

"""
Export ONNX model from PyTorch checkpoint for 480x170 input size.
Usage: python scripts/export_onnx.py <checkpoint_path> <output_path>
"""
import sys
import torch
import torch.onnx
import numpy as np

if len(sys.argv) < 3:
    print("Usage: python scripts/export_onnx.py <checkpoint_path> <output_onnx_path>")
    print("Example: python scripts/export_onnx.py checkpoints/model.pth deployment/models/type_classifier_480x170.onnx")
    sys.exit(1)

checkpoint_path = sys.argv[1]
output_path = sys.argv[2]

# Load checkpoint
print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract model architecture and weights
# NOTE: This assumes your checkpoint has 'model' or 'state_dict' key
# Adjust based on your actual checkpoint structure
if 'model' in checkpoint:
    model = checkpoint['model']
elif 'state_dict' in checkpoint:
    # If only state_dict, you'll need to reconstruct the model
    # This is a placeholder - adjust based on your actual model architecture
    print("Warning: Only state_dict found. You need to define your model architecture first.")
    print("Example:")
    print("  from your_model import YourModel")
    print("  model = YourModel()")
    print("  model.load_state_dict(checkpoint['state_dict'])")
    sys.exit(1)
else:
    model = checkpoint

model.eval()

# Create dummy input (B, C, H, W) = (1, 3, 170, 480)
# Note: ONNX expects (batch, channels, height, width)
dummy_input = torch.randn(1, 3, 170, 480)

print(f"Exporting to {output_path}...")
print(f"Input shape: {dummy_input.shape} (batch, channels, height, width)")

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11,  # Adjust if needed
    do_constant_folding=True,
)

print(f"✓ Exported ONNX model to {output_path}")
print(f"  Expected input: (batch, 3, 170, 480)")
print(f"  Output: (batch, num_classes)")

