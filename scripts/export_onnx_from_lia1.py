#!/usr/bin/env python3
# scripts/export_onnx_from_lia1.py

"""
Export ONNX model from lia1test checkpoint with 480x170 input size.
Usage: python scripts/export_onnx_from_lia1.py <checkpoint_path> <output_path> [backbone]
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

# Import model definition from lia1test
lia1_path = Path("/c/Users/olabl/Documents/GitHub/lia1test")
if not lia1_path.exists():
    # Try Windows path format
    lia1_path = Path(r"C:\Users\olabl\Documents\GitHub\lia1test")

sys.path.insert(0, str(lia1_path))
# Change to lia1 directory to resolve relative imports
import os
old_cwd = os.getcwd()
os.chdir(str(lia1_path))
try:
    from src.train.type_train import get_model
finally:
    os.chdir(old_cwd)

def export_onnx(checkpoint_path: str, output_path: str, backbone: str = "resnet18", num_classes: int = 3):
    """Export model to ONNX with 170x480 input size"""
    
    # Load model
    print(f"Loading model: {backbone} with {num_classes} classes...")
    model = get_model(backbone, num_classes)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy input with correct shape: (batch, channels, height, width) = (1, 3, 170, 480)
    dummy_input = torch.randn(1, 3, 170, 480)
    
    print(f"Exporting to {output_path}...")
    print(f"Input shape: {dummy_input.shape} (batch, channels, height, width)")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # Static shape for now
    )
    
    print(f"✓ Exported ONNX model to {output_path}")
    print(f"  Expected input: (batch, 3, 170, 480)")
    print(f"  Output: (batch, {num_classes})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/export_onnx_from_lia1.py <checkpoint_path> <output_path> [backbone]")
        print("Example: python scripts/export_onnx_from_lia1.py ../lia1test/archive/checkpoints/best_type_model.pth deployment/models/type_classifier_480x170.onnx resnet18")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2]
    backbone = sys.argv[3] if len(sys.argv) > 3 else "resnet18"
    
    export_onnx(checkpoint_path, output_path, backbone)

