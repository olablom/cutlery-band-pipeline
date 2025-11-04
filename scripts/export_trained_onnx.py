#!/usr/bin/env python3
# scripts/export_trained_onnx.py

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset_480x170 import CLASS_MAP

CKPT_PATH = "checkpoints/best_resnet18_480x170.pth"
OUT_PATH = "deployment/models/type_classifier_480x170.onnx"

def build_model():
    m = models.resnet18(weights=None)
    m.conv1.stride = (1, 1)
    m.maxpool.stride = (1, 1)
    m.fc = nn.Linear(m.fc.in_features, len(CLASS_MAP))
    return m

def main():
    model = build_model()
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 170, 480)
    
    torch.onnx.export(
        model,
        dummy,
        OUT_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
    )
    
    print("exported to", OUT_PATH)

if __name__ == "__main__":
    main()

