#!/usr/bin/env python3
# src/dataset_480x170.py

from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset

CLASS_MAP = {
    "fork": 0,
    "knife": 1,
    "spoon": 2,
}

def list_samples(root_dir: str):
    root = Path(root_dir)
    samples = []
    for cls in CLASS_MAP.keys():
        for p in (root / cls).rglob("*.jpg"):
            rel = p.relative_to(root).as_posix()
            samples.append(rel)
    return samples

class CutleryDataset(Dataset):
    def __init__(self, root_dir: str, samples, transform=None):
        self.root_dir = Path(root_dir)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel = self.samples[idx]
        img_path = self.root_dir / rel
        img = Image.open(img_path).convert("RGB")
        cls_name = rel.split("/")[0]
        label = CLASS_MAP[cls_name]

        if self.transform:
            img = self.transform(img)

        return img, label

