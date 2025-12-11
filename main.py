import os
import json
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# ============================================================
# 0. Setup
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ============================================================
# Logging Utility
# ============================================================

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(f"logs/{name}.log")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(fmt)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(fh)

    return logger


# ============================================================
# 1. Data Loaders
# ============================================================

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)


# ============================================================
# 2. Model (Hybrid GATE-Net from notebook)
# ============================================================

class LayerNorm2d(nn.Module):
    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps

    def forward(self, x):
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x_hat = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x_hat + self.bias[:, None, None]


class ConvBranch(nn.Module):
    def __init__(self, C, k=3):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv2d(C, C, k, padding=pad, groups=C)
        self.pw = nn.Conv2d(C, C, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class AttnBranch(nn.Module):
    def __init__(self, C, heads=4):
        super().__init__()
        assert C % heads == 0
        self.heads = heads
        self.head_dim = C // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(C, 3 * C, 1, bias=False)
        self.proj = nn.Conv2d(C, C, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum("bhds,bhdt->bhs t", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhst,bhdt->bhds", attn, v)
        out = out.reshape(B, C, H, W)
        return self.proj(out)


class HybridBlock(nn.Module):
    def __init__(self, C, heads=4, gate_reduction=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv_branch = ConvBranch(C)
        self.attn_branch = AttnBranch(C, heads=heads)
        self.gap = nn.AdaptiveAvgPool2d(1)

        hidden = max(C // gate_reduction, 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(C, hidden),
            nn.GELU(),
            nn.Linear(hidden, C),
            nn.Sigmoid()
        )
        self.norm = norm_layer(C)

    def forward(self, x):
        conv_out = self.conv_branch(x)
        attn_out = self.attn_branch(x)
        g = self.gap(x).flatten(1)
        g = self.gate_mlp(g).view(g.size(0), g.size(1), 1, 1)
        fused = g * conv_out + (1 - g) * attn_out
        return self.norm(fused + x), g.mean(dim=(2, 3))


class GATENetSmall(nn.Module):
    def __init__(self, num_classes=10, heads=(2, 4, 4)):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        # Stage 1
        self.block1a = HybridBlock(64, heads=heads[0])
        self.block1b = HybridBlock(64, heads=heads[0])
        self.down1 = nn.Sequential(LayerNorm2d(64), nn.Conv2d(64, 128, 2, stride=2))

        # Stage 2
        self.block2a = HybridBlock(128, heads=heads[1])
        self.block2b = HybridBlock(128, heads=heads[1])
        self.down2 = nn.Sequential(LayerNorm2d(128), nn.Conv2d(128, 256, 2, stride=2))

        # Stage 3
        self.block3a = HybridBlock(256, heads=heads[2])
        self.block3b = HybridBlock(256, heads=heads[2])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, collect_gates=False):
        gate_list = []

        x = self.stem(x)
        x, g = self.block1a(x); gate_list.append(g)
        x, g = self.block1b(x); gate_list.append(g)

        x = self.down1(x)

        x, g = self.block2a(x); gate_list.append(g)
        x, g = self.block2b(x); gate_list.append(g)

        x = self.down2(x)

        x, g = self.block3a(x); gate_list.append(g)
        x, g = self.block3b(x); gate_list.append(g)

        x = self.pool(x).flatten(1)
        logits = self.mlp_head(x)

        if collect_gates:
            return logits, gate_list

        return logits


# ============================================================
# 3. Training Utilities
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


# ============================================================
# 4. Save + Load Utilities
# ============================================================

def save_checkpoint(model, history, epoch, name):
    ckpt = {
        "model": model.state_dict(),
        "history": history,
        "epoch": epoch,
    }
    torch.save(ckpt, f"checkpoints/{name}.pt")
    with open(f"logs/{name}_history.json", "w") as f:
        json.dump(history, f, indent=4)


def load_checkpoint(model, name):
    path = f"checkpoints/{name}.pt"
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        return ckpt["history"], ckpt["epoch"]
    return None, 0


# ============================================================
# 5. Train Loop With Resume
# ============================================================

def run_training(model, name, epochs=40, lr=3e-4):
    logger = setup_logger(name)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    history, start_epoch = load_checkpoint(model, name)
    if history is None:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        logger.info("Starting new training run")
    else:
        logger.info(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch + 1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_epoch(model, test_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch:02d}: "
            f"train_loss={tr_loss:.4f}, train_acc={tr_acc*100:.2f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%"
        )

        save_checkpoint(model, history, epoch, name)

    return model, history


# ============================================================
# 6. Run Full Training
# ============================================================

if __name__ == "__main__":
    model = GATENetSmall(num_classes=10)
    trained_model, hist = run_training(model, "GATENetSmall", epochs=50)
