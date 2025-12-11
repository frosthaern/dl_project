# ============================================================
# main.py – Train CNN, ViT, Hybrid(CNN+ViT) and store metrics
# ============================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ============================================================
# 0. Setup
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

os.makedirs("results", exist_ok=True)

# ============================================================
# 1. CIFAR-10 Data Loaders
# ============================================================
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

# ============================================================
# 2. Baseline CNN
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ============================================================
# 3. Very Simple ViT Baseline
# ============================================================
class TinyViT(nn.Module):
    def __init__(self, num_classes=10, dim=128, depth=4, heads=4, patch=4):
        super().__init__()
        self.patch = patch
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        num_patches = (32 // patch) ** 2

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                  # B,dim,H',W'
        x = x.flatten(2).transpose(1, 2)         # B,N,dim
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# ============================================================
# 4. Hybrid CNN + ViT
# ============================================================
class HybridCNNViT(nn.Module):
    """
    CNN backbone → produces tokens → ViT → classifier
    Ensures stronger representation, tends to outperform baselines.
    """
    def __init__(self, num_classes=10, dim=192, depth=4, heads=6):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU
