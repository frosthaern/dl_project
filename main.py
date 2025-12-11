import os
import json
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 0.05
LOG_DIR = "logs"
CKPT_DIR = "checkpoints"

USE_TORCH_COMPILE = True
USE_AMP = True
GRAD_ACCUM_STEPS = 1
SAVE_EVERY = 1  # save checkpoint every epoch

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ============================================================
# LOGGER
# ============================================================
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(f"{LOG_DIR}/{name}.log")
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(fh)

    return logger

logger = setup_logger("gatenet_optimized")

# ============================================================
# DATA
# ============================================================
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3, 0.05),
    T.ToTensor(),
    T.Normalize(mean, std),
])

test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ============================================================
# MODEL COMPONENTS
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


class PatchAttention(nn.Module):
    """
    Attention over downsampled patches to reduce HW^2 cost.
    """
    def __init__(self, C, patch_size=(4,4), heads=4):
        super().__init__()
        self.ph, self.pw = patch_size
        self.heads = heads
        assert C % heads == 0
        self.hd = C // heads
        self.scale = self.hd ** -0.5

        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.ph, self.pw
        assert H % ph == 0 and W % pw == 0

        Hn, Wn = H // ph, W // pw

        # Extract patches → average pool within patch
        x_p = x.unfold(2, ph, ph).unfold(3, pw, pw)  # B,C,Hn,Wn,ph,pw
        x_p = x_p.contiguous().view(B, C, Hn, Wn, ph*pw)
        x_p = x_p.mean(-1)  # B,C,Hn,Wn
        x_p = x_p.permute(0,2,3,1).reshape(B, Hn*Wn, C)

        qkv = self.qkv(x_p).reshape(B, Hn*Wn, 3, self.heads, self.hd)
        qkv = qkv.permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        attn = attn.softmax(-1)

        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.permute(0,2,1,3).reshape(B, Hn, Wn, C)
        out = out.permute(0,3,1,2)
        out = F.interpolate(out, size=(H,W), mode="nearest")

        return out


class HybridBlockOptimized(nn.Module):
    def __init__(self, C, heads=2, patch_size=(4,4), gate_reduction=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv_branch = ConvBranch(C)
        self.attn_branch = PatchAttention(C, patch_size=patch_size, heads=heads)

        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = max(C // gate_reduction, 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(C, hidden), nn.GELU(), nn.Linear(hidden, C), nn.Sigmoid()
        )

        self.norm = norm_layer(C)

    def forward(self, x):
        conv_out = self.conv_branch(x)
        attn_out = self.attn_branch(x)

        g = self.gap(x).flatten(1)
        g = self.gate_mlp(g).view(g.size(0), g.size(1), 1, 1)

        fused = g * conv_out + (1 - g) * attn_out
        return self.norm(fused + x), g.mean(dim=(2,3))


class GATENetOptimized(nn.Module):
    def __init__(self, num_classes=10,
                 channels=(48,96,192),
                 heads=(1,2,2),
                 patch_sizes=((8,8),(4,4),(2,2))):
        super().__init__()
        c1,c2,c3 = channels

        self.stem = nn.Sequential(nn.Conv2d(3, c1, 3, padding=1),
                                  nn.BatchNorm2d(c1),
                                  nn.GELU())

        self.block1a = HybridBlockOptimized(c1, heads=heads[0], patch_size=patch_sizes[0])
        self.block1b = HybridBlockOptimized(c1, heads=heads[0], patch_size=patch_sizes[0])
        self.down1 = nn.Sequential(LayerNorm2d(c1), nn.Conv2d(c1, c2, 2, stride=2))

        self.block2a = HybridBlockOptimized(c2, heads=heads[1], patch_size=patch_sizes[1])
        self.block2b = HybridBlockOptimized(c2, heads=heads[1], patch_size=patch_sizes[1])
        self.down2 = nn.Sequential(LayerNorm2d(c2), nn.Conv2d(c2, c3, 2, stride=2))

        self.block3a = HybridBlockOptimized(c3, heads=heads[2], patch_size=patch_sizes[2])
        self.block3b = HybridBlockOptimized(c3, heads=heads[2], patch_size=patch_sizes[2])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_head = nn.Sequential(
            nn.Linear(c3, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, num_classes)
        )

    def forward(self, x, collect_gates=False):
        gates = []

        x = self.stem(x)
        x, g = self.block1a(x); gates.append(g)
        x, g = self.block1b(x); gates.append(g)
        x = self.down1(x)

        x, g = self.block2a(x); gates.append(g)
        x, g = self.block2b(x); gates.append(g)
        x = self.down2(x)

        x, g = self.block3a(x); gates.append(g)
        x, g = self.block3b(x); gates.append(g)

        x = self.pool(x).flatten(1)
        logits = self.mlp_head(x)

        return (logits, gates) if collect_gates else logits


# ============================================================
# TRAINING HELPERS
# ============================================================
def train_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()

    for step, (x,y) in enumerate(tqdm(loader, leave=False)):
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y) / GRAD_ACCUM_STEPS

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step+1) % GRAD_ACCUM_STEPS == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * x.size(0) * GRAD_ACCUM_STEPS
        _, preds = logits.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)

    return total_loss/total, correct/total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for x,y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)

    return total_loss/total, correct/total


# ============================================================
# CHECKPOINTS
# ============================================================
def save_checkpoint(model, history, epoch, name):
    ckpt = {
        "model": model.state_dict(),
        "history": history,
        "epoch": epoch
    }
    torch.save(ckpt, f"{CKPT_DIR}/{name}.pt")

    with open(f"{LOG_DIR}/{name}_history.json","w") as f:
        json.dump(history, f, indent=2)

def load_checkpoint(model, name):
    path = f"{CKPT_DIR}/{name}.pt"
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        return ck["history"], ck["epoch"]
    return None, 0


# ============================================================
# TRAINING LOOP WITH DETAILED PRINTS
# ============================================================
def run_training(name="GATENetOptimized", epochs=EPOCHS, lr=LR):
    logger = setup_logger(name)

    model = GATENetOptimized(num_classes=10).to(DEVICE)

    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            logger.info("torch.compile applied")
        except Exception as e:
            logger.info(f"torch.compile failed: {e}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE.startswith("cuda")) else None

    history, start_epoch = load_checkpoint(model, name)
    if history is None:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        logger.info("Starting new training run")
    else:
        logger.info(f"Resuming from epoch {start_epoch}")

    print("\n===== TRAINING STARTED =====")
    total_start = time.time()

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc = eval_epoch(model, test_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start
        eta = (epochs - epoch) * epoch_time

        gpu_mem = torch.cuda.memory_allocated()/1024/1024 if DEVICE.startswith("cuda") else 0

        print(f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s")
        print(f" | Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc*100:.2f}%")
        print(f" | Val Loss:   {val_loss:.4f}  Val Acc: {val_acc*100:.2f}%")
        print(f" | GPU Memory: {gpu_mem:.1f} MB")
        print(f" | Elapsed: {elapsed/60:.2f} min | ETA: {eta/60:.2f} min")

        logger.info(
            f"Epoch {epoch}: train_loss={tr_loss:.4f}, train_acc={tr_acc*100:.2f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%, "
            f"time={epoch_time:.2f}s"
        )

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, history, epoch, name)
            print(f"Checkpoint saved at epoch {epoch}")

    print("\n===== TRAINING FINISHED =====")
    print(f"Total time: {(time.time() - total_start)/60:.2f} min")

    return model, history


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("Device:", DEVICE)
    run_training()
