#!/usr/bin/env python3
# src/train_480x170.py

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from src.dataset_480x170 import CLASS_MAP, CutleryDataset, list_samples

DATA_DIR = "dataset/processed"
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 480, 170
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
VAL_SPLIT = 0.15

def make_splits(samples):
    random.shuffle(samples)
    n_val = int(len(samples) * VAL_SPLIT)
    return samples[n_val:], samples[:n_val]

def get_transforms():
    train_tf = transforms.Compose([
        transforms.ToTensor(),              # redan 480x170
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    return train_tf, val_tf

def build_model(num_classes=3):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # 170 är lågt → behåll mer spatial info
    m.conv1.stride = (1, 1)
    m.maxpool.stride = (1, 1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def main():
    all_samples = list_samples(DATA_DIR)
    train_samples, val_samples = make_splits(all_samples)
    
    train_tf, val_tf = get_transforms()
    
    train_ds = CutleryDataset(DATA_DIR, train_samples, transform=train_tf)
    val_ds = CutleryDataset(DATA_DIR, val_samples, transform=val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = build_model(len(CLASS_MAP)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_val = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optim.step()
            
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            tot += x.size(0)
        
        train_loss = loss_sum / tot
        train_acc = correct / tot
        
        # val
        model.eval()
        v_tot, v_corr = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                v_corr += (pred == y).sum().item()
                v_tot += x.size(0)
        
        val_acc = v_corr / v_tot
        
        print(f"epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), CKPT_DIR / "best_resnet18_480x170.pth")
            print("saved best")
    
    print("done. best val acc:", best_val)

if __name__ == "__main__":
    main()

