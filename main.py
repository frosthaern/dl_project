# main.py
# Train: SimpleCNN, TinyViT, ConvNeXtSmall+ViT hybrid on CIFAR-10
# Saves: results/<model>_metrics.json, <model>_plot.png, comparison_plot.png, checkpoints/
# Requirements: torch, torchvision, matplotlib, tqdm

import os
import json
import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = Path("results")
OUTDIR.mkdir(exist_ok=True)
(OUTDIR / "checkpoints").mkdir(exist_ok=True)
NUM_CLASSES = 10
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 80                   # adjust for compute budget (40-120 recommended)
LR = 3e-4
WEIGHT_DECAY = 0.05
SEED = 42

torch.manual_seed(SEED)

# -------------------------
# Data: CIFAR-10 with Augmentations
# -------------------------
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
    T.ToTensor(),
    T.Normalize(mean, std),
    T.RandomErasing(p=0.25, scale=(0.02,0.33))
])

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std)
])

train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# -------------------------
# Models
# -------------------------
# 1) Simple baseline CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)             # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 2) Tiny ViT baseline
class TinyViT(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dim=128, depth=6, heads=4, patch=4):
        super().__init__()
        self.patch = patch
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)  # 32->8 with patch=4
        num_patches = (32 // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                  # B,dim,H',W'
        x = x.flatten(2).transpose(1,2)          # B, N, dim
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.head(x[:, 0])

# 3) ConvNeXt-small + ViT-small hybrid (strong model)
# ConvNeXt block (lightweight)
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4*dim, dim)

    def forward(self, x):
        # x: B, C, H, W
        residual = x
        x = self.dw(x)                         # B,C,H,W
        x = x.permute(0,2,3,1)                 # B,H,W,C
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x.permute(0,3,1,2)                 # B,C,H,W
        return x + residual

class ConvNeXtSmallViTHybrid(nn.Module):
    """
    ConvNeXt-small backbone (stem -> blocks) -> tokens -> ViT encoder -> CLS head.
    Designed to be stronger than TinyViT for CIFAR-10.
    """
    def __init__(self, num_classes=NUM_CLASSES, dim=192, conv_blocks=6, vit_depth=6, vit_heads=6):
        super().__init__()
        self.dim = dim
        # Stem: downsample 32->8 directly (stride=4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=4, stride=4),  # 32 -> 8
            nn.LayerNorm(dim, eps=1e-6)                  # note: LayerNorm expects channels-last - handled in blocks
        )
        # A few ConvNeXt blocks (operate in channels-first; each block handles permute)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(conv_blocks)])

        # Tokenization already at 8x8 (64 tokens). Optionally reduce tokens via 2x2 conv to 4x4 tokens.
        # We'll keep 8x8 tokens but project to dim if needed (already dim).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # ViT encoder over tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=vit_heads, dim_feedforward=dim*4, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=vit_depth)
        self.head = nn.Linear(dim, num_classes)

        # small init
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        # stem expects N,C,H,W -> after Conv we still have channels-first
        x = self.stem[0](x)   # conv
        # conv produced B,dim,8,8; apply remaining norm manually to channels-last for stability inside blocks
        for block in self.blocks:
            x = block(x)
        # flatten tokens
        x = x.flatten(2).transpose(1, 2)  # B, N=64, dim
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)    # B, 1+N, dim
        x = self.transformer(x)
        return self.head(x[:, 0])

# -------------------------
# Utilities: training, eval, save
# -------------------------
def accuracy_from_logits(logits, labels):
    _, preds = logits.max(1)
    return preds.eq(labels).sum().item() / labels.size(0)

def save_metrics(metrics: dict, name: str):
    with open(OUTDIR / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f)

def plot_metrics(hist: dict, name: str):
    plt.figure(figsize=(6,4))
    plt.plot(hist["train_acc"], label="train_acc")
    plt.plot(hist["val_acc"], label="val_acc")
    plt.title(name + " accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{name}_plot.png")
    plt.close()

# -------------------------
# Train loop (with amp)
# -------------------------
def train_one(model, optimizer, criterion, loader, scaler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(xb)
            loss = criterion(logits, yb)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = xb.size(0)
        running_loss += loss.item() * bs
        running_acc += (logits.argmax(1) == yb).sum().item()
        total += bs
    return running_loss / total, running_acc / total

@torch.no_grad()
def validate_one(model, criterion, loader):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.size(0)
        running_loss += loss.item() * bs
        running_acc += (logits.argmax(1) == yb).sum().item()
        total += bs
    return running_loss / total, running_acc / total

# -------------------------
# Runner for a single model
# -------------------------
def run_training(model, name, epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY, use_amp=True, save_best=True):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # cosine LR with warmup
    total_steps = epochs * len(train_loader)
    def lr_lambda(step):
        # linear warmup for first 5% steps then cosine anneal
        warmup_steps = max(1, int(0.05 * total_steps))
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = torch.cuda.amp.GradScaler() if (use_amp and DEVICE == "cuda") else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0

    step = 0
    for ep in range(1, epochs + 1):
        train_loss, train_acc = train_one(model, optimizer, criterion, train_loader, scaler=scaler)
        val_loss, val_acc = validate_one(model, criterion, test_loader)
        # step scheduler by steps-per-epoch
        for _ in range(len(train_loader)):
            scheduler.step()
            step += 1

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{name}] Epoch {ep:03d}/{epochs}  train_acc={train_acc*100:5.2f}%  val_acc={val_acc*100:5.2f}%  val_loss={val_loss:.4f}")

        # save best
        if save_best and val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": ep,
                "val_acc": val_acc
            }
            torch.save(ckpt, OUTDIR / "checkpoints" / f"{name}_best.pt")

    save_metrics(history, name)
    plot_metrics(history, name)
    return history

# -------------------------
# Full experiment: train three models sequentially
# -------------------------
def main():
    # 1. SimpleCNN
    cnn = SimpleCNN()
    cnn_hist = run_training(cnn, "SimpleCNN", epochs=EPOCHS//2)  # smaller budget for baseline
    # 2. TinyViT
    vit = TinyViT(dim=160, depth=8, heads=5, patch=4)
    vit_hist = run_training(vit, "TinyViT", epochs=EPOCHS)
    # 3. ConvNeXtSmall + ViT Hybrid (strong)
    hybrid = ConvNeXtSmallViTHybrid(dim=224, conv_blocks=8, vit_depth=8, vit_heads=7)
    hybrid_hist = run_training(hybrid, "ConvNeXtSmall_ViT", epochs=EPOCHS, lr=2e-4)

    # comparison plot
    plt.figure(figsize=(7,5))
    # align lengths by padding last value if needed
    def pad(lst, L):
        return lst + [lst[-1]]*(L - len(lst))
    max_len = max(len(cnn_hist["val_acc"]), len(vit_hist["val_acc"]), len(hybrid_hist["val_acc"]))
    plt.plot(pad(cnn_hist["val_acc"], max_len), label="SimpleCNN")
    plt.plot(pad(vit_hist["val_acc"], max_len), label="TinyViT")
    plt.plot(pad(hybrid_hist["val_acc"], max_len), label="ConvNeXtSmall+ViT", linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel("val accuracy")
    plt.title("Model comparison (val acc)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "comparison_plot.png")
    plt.close()

    print("Training finished. Results in:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
