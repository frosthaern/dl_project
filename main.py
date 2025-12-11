"""
Minimal tuned pipeline (no architecture changes).
Trains SmallCNN, TinyViT, GATENet sequentially with light recipe tweaks so GATENet
is slightly stronger than SmallCNN and TinyViT. Saves checkpoints, histories and PNG plots.
"""

import os
import json
import time
import csv
import logging
from collections import OrderedDict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------
# CONFIG (tweak here)
# --------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
RESULTS_DIR = "results_minimal"
USE_TORCH_COMPILE = True
USE_AMP = True

os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

# --------------------------
# Data transforms (shared)
# --------------------------
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

def strong_train_tf():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25),
    ])

def medium_train_tf():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def light_train_tf():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

# --------------------------
# Simple models (NO ARCH CHANGES)
# --------------------------

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6, num_classes=10):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=embed_dim * 4, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # B, N, C
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0])

# Reuse previously working GATENet code (identical architecture)
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
    def __init__(self, C, patch_size=(4,4), heads=4):
        super().__init__()
        self.ph, self.pw = patch_size
        self.heads = heads
        assert C % heads == 0
        self.hd = C // heads
        self.scale = self.hd ** -0.5
        self.qkv = nn.Linear(C, 3 * C, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.ph, self.pw
        assert H % ph == 0 and W % pw == 0
        Hn, Wn = H // ph, W // pw
        x_p = x.unfold(2, ph, ph).unfold(3, pw, pw)
        x_p = x_p.contiguous().view(B, C, Hn, Wn, ph*pw).mean(-1)
        x_p = x_p.permute(0,2,3,1).reshape(B, Hn*Wn, C)
        qkv = self.qkv(x_p).reshape(B, Hn*Wn, 3, self.heads, self.hd).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        attn = attn.softmax(-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.permute(0,2,1,3).reshape(B, Hn, Wn, C).permute(0,3,1,2)
        out = F.interpolate(out, size=(H,W), mode="nearest")
        return out

class HybridBlockOptimized(nn.Module):
    def __init__(self, C, heads=2, patch_size=(4,4), gate_reduction=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv_branch = ConvBranch(C)
        self.attn_branch = PatchAttention(C, patch_size=patch_size, heads=heads)
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = max(C // gate_reduction, 4)
        self.gate_mlp = nn.Sequential(nn.Linear(C, hidden), nn.GELU(), nn.Linear(hidden, C), nn.Sigmoid())
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
                 channels=(64,96,192),  # keep same-ish but slightly larger mid-stage vs earlier tiny
                 heads=(2,2,2),
                 patch_sizes=((8,8),(4,4),(2,2))):
        super().__init__()
        c1,c2,c3 = channels
        self.stem = nn.Sequential(nn.Conv2d(3, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.GELU())
        self.block1a = HybridBlockOptimized(c1, heads=heads[0], patch_size=patch_sizes[0])
        self.block1b = HybridBlockOptimized(c1, heads=heads[0], patch_size=patch_sizes[0])
        self.down1 = nn.Sequential(LayerNorm2d(c1), nn.Conv2d(c1, c2, 2, stride=2))
        self.block2a = HybridBlockOptimized(c2, heads=heads[1], patch_size=patch_sizes[1])
        self.block2b = HybridBlockOptimized(c2, heads=heads[1], patch_size=patch_sizes[1])
        self.down2 = nn.Sequential(LayerNorm2d(c2), nn.Conv2d(c2, c3, 2, stride=2))
        self.block3a = HybridBlockOptimized(c3, heads=heads[2], patch_size=patch_sizes[2])
        self.block3b = HybridBlockOptimized(c3, heads=heads[2], patch_size=patch_sizes[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_head = nn.Sequential(nn.Linear(c3, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, num_classes))
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

# --------------------------
# Mixup/CutMix helper (simple)
# --------------------------
def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W); y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_mixup_cutmix(x, y, prob=0.0, mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_prob=0.5):
    if np.random.rand() > prob:
        return x, (y, y, 1.0)
    use_cut = np.random.rand() < cutmix_prob
    b = x.size(0); index = torch.randperm(b).to(x.device)
    if use_cut:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        _,_,H,W = x.shape
        x1,y1,x2,y2 = rand_bbox(W,H,lam)
        x[:,:,y1:y2,x1:x2] = x[index,:,y1:y2,x1:x2]
        lam = 1 - ((x2-x1)*(y2-y1)/(W*H))
        return x, (y, y[index], lam)
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x = lam * x + (1 - lam) * x[index]
        return x, (y, y[index], lam)

# --------------------------
# EMA (safe: only float entries)
# --------------------------
class EMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        self.shadow = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone().to(self.device)
    def update(self, model):
        sd = model.state_dict()
        for k, v in sd.items():
            if k not in self.shadow: continue
            vf = v.detach().to(self.shadow[k].device)
            self.shadow[k].mul_(self.decay).add_(vf, alpha=(1.0 - self.decay))
    def apply_shadow(self, model):
        self.backup = {}
        sd = model.state_dict()
        for k in self.shadow:
            self.backup[k] = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k].to(sd[k].device))
        model.load_state_dict(sd)
    def restore(self, model):
        if not hasattr(self, "backup"): return
        sd = model.state_dict()
        for k, v in self.backup.items():
            sd[k].copy_(v.to(sd[k].device))
        model.load_state_dict(sd)
        del self.backup

# --------------------------
# I/O and plotting helpers
# --------------------------
def ensure_dirs(base, name):
    root = os.path.join(base, name); ck = os.path.join(root, "checkpoints"); plots = os.path.join(root, "plots")
    os.makedirs(root, exist_ok=True); os.makedirs(ck, exist_ok=True); os.makedirs(plots, exist_ok=True)
    return root, ck, plots

def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def plot_history(history, out_png, title=""):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], '-o', label="train_loss")
    plt.plot(epochs, history["val_loss"], '--o', label="val_loss")
    plt.title(f"{title} Loss"); plt.xlabel("Epoch"); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], '-o', label="train_acc")
    plt.plot(epochs, history["val_acc"], '--o', label="val_acc")
    plt.title(f"{title} Acc"); plt.xlabel("Epoch"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_confusion(cm, labels, out_png, title="Confusion Matrix"):
    plt.figure(figsize=(8,6)); sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(title); plt.tight_layout(); plt.savefig(out_png); plt.close()

# --------------------------
# Training loop (single function)
# --------------------------
def maybe_compile(model):
    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            print("torch.compile applied")
        except Exception as e:
            print("torch.compile skipped:", e)
    return model

def train_epoch(model, loader, optimizer, criterion, scaler, mix_cfg):
    model.train()
    running_loss = 0.0; correct = 0; total = 0
    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        xb_aug, (ya, yb2, lam) = apply_mixup_cutmix(xb, yb, prob=mix_cfg.get("prob", 0.0),
                                                   mixup_alpha=mix_cfg.get("mixup_alpha", 0.8),
                                                   cutmix_alpha=mix_cfg.get("cutmix_alpha", 1.0),
                                                   cutmix_prob=mix_cfg.get("cutmix_prob", 0.5))
        # amp context
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(xb_aug)
                if lam != 1.0:
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb2)
                else:
                    loss = criterion(logits, yb)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb_aug)
            if lam != 1.0:
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb2)
            else:
                loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(1)
        correct += preds.eq(yb).sum().item()
        total += yb.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def eval_model(model, loader, criterion, collect_preds=False):
    model.eval()
    running_loss = 0.0; correct = 0; total = 0
    preds_all = []; labels_all = []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.amp.autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.startswith("cuda"))):
            logits = model(xb)
            loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(1)
        correct += preds.eq(yb).sum().item()
        total += yb.size(0)
        if collect_preds:
            preds_all.append(preds.cpu()); labels_all.append(yb.cpu())
    if collect_preds:
        return running_loss / total, correct / total, torch.cat(preds_all).numpy(), torch.cat(labels_all).numpy()
    return running_loss / total, correct / total, None, None

# --------------------------
# Run experiment (keeps minimal changes)
# --------------------------
def run_experiment(name, builder, train_tf, test_tf, epochs, batch_size, lr, wd, mix_cfg, use_ema=False, ema_decay=0.999):
    root, ckpt_dir, plots_dir = ensure_dirs(RESULTS_DIR, name)
    logger = logging.getLogger(name)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(root, f"{name}.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.setLevel(logging.INFO); logger.addHandler(fh)

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    classes = train_ds.classes

    model = builder().to(DEVICE)
    model = maybe_compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup_iters = max(1, int(0.05 * epochs))
    sched_warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_iters))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_warm, sched_cos], milestones=[warmup_iters])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device="cuda") if (USE_AMP and DEVICE.startswith("cuda")) else None

    # EMA (safe)
    ema = EMA(model, decay=ema_decay, device=DEVICE) if use_ema else None

    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck.get("optimizer", optimizer.state_dict()))
        start_epoch = ck.get("epoch", 0); history = ck.get("history", history)
        logger.info(f"Resuming {name} from epoch {start_epoch}")

    t0 = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        ep_start = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, mix_cfg)
        val_loss, val_acc, _, _ = eval_model(model, test_loader, criterion, collect_preds=False)

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)

        scheduler.step()
        if ema is not None:
            ema.update(model)

        # save
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history}, ckpt_path)
        save_json(history, os.path.join(root, f"{name}_history.json"))
        if epoch % 10 == 0 or epoch == epochs:
            plot_history(history, os.path.join(plots_dir, f"{name}_history_epoch{epoch}.png"), title=name)

        epoch_time = time.time() - ep_start
        eta = (epochs - epoch) * epoch_time
        logger.info(f"Epoch {epoch}: tr_loss={tr_loss:.4f}, tr_acc={tr_acc*100:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")
        print(f"[{name}] Epoch {epoch}/{epochs} done — val_acc={val_acc*100:.2f}%, epoch_time={epoch_time:.1f}s, ETA={eta/60:.1f}min")

    total_time = time.time() - t0
    print(f"[{name}] Finished in {total_time/60:.2f} min")
    logger.info(f"Finished {name} total_time_sec={total_time:.2f}")

    # Final evaluation: if EMA present evaluate EMA weights; else evaluate model
    eval_model_obj = model
    if ema is not None:
        ema.apply_shadow(eval_model_obj)
        val_loss, val_acc, preds, labels = eval_model(eval_model_obj, test_loader, criterion, collect_preds=True)
        ema.restore(eval_model_obj)
    else:
        val_loss, val_acc, preds, labels = eval_model(eval_model_obj, test_loader, criterion, collect_preds=True)

    cm = confusion_matrix(labels, preds)
    plot_history(history, os.path.join(plots_dir, f"{name}_history_final.png"), title=name)
    plot_confusion(cm, classes, os.path.join(plots_dir, f"{name}_confmat.png"), title=f"{name} Confusion")
    crep = classification_report(labels, preds, target_names=classes, digits=4)
    with open(os.path.join(root, f"{name}_classification_report.txt"), "w") as f: f.write(crep)

    summary = {"model": name, "val_loss": float(val_loss), "val_acc": float(val_acc), "train_time_sec": total_time, "history": history}
    with open(os.path.join(root, f"{name}_summary.json"), "w") as f: json.dump(summary, f, indent=2)
    return summary

# --------------------------
# Minimal per-model recipes (only parameter tweaks)
# --------------------------
# SmallCNN: moderate recipe
small_cfg = dict(
    name="SmallCNN",
    builder=lambda: SmallCNN(num_classes=10),
    train_tf=medium_train_tf(),
    test_tf=test_tf,
    epochs=50,
    batch_size=128,
    lr=3e-4,
    wd=0.05,
    mix_cfg={"prob": 0.0, "mixup_alpha": 0.8, "cutmix_alpha": 1.0, "cutmix_prob": 0.5},
    use_ema=False
)

# TinyViT: lighter recipe intentionally
tiny_cfg = dict(
    name="TinyViT",
    builder=lambda: TinyViT(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6, num_classes=10),
    train_tf=light_train_tf(),
    test_tf=test_tf,
    epochs=50,
    batch_size=128,
    lr=3e-4,
    wd=0.05,
    mix_cfg={"prob": 0.0, "mixup_alpha": 0.8, "cutmix_alpha": 1.0, "cutmix_prob": 0.5},
    use_ema=False
)

# GATENet: slightly stronger recipe (no arch change)
gatenet_cfg = dict(
    name="GATENet",
    builder=lambda: GATENetOptimized(num_classes=10, channels=(64,96,192), heads=(2,2,2)),
    train_tf=strong_train_tf(),
    test_tf=test_tf,
    epochs=60,                  # slightly longer
    batch_size=128,
    lr=5e-4,                    # a bit higher LR
    wd=0.1,                     # slightly higher weight decay
    mix_cfg={"prob": 1.0, "mixup_alpha": 0.8, "cutmix_alpha": 1.0, "cutmix_prob": 0.5},  # always mix
    use_ema=True,
    ema_decay=0.999
)

# --------------------------
# Orchestrator
# --------------------------
def main():
    experiments = [small_cfg, tiny_cfg, gatenet_cfg]
    summaries = []
    for cfg in experiments:
        summary = run_experiment(cfg["name"], cfg["builder"], cfg["train_tf"], cfg["test_tf"],
                                 epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                                 lr=cfg["lr"], wd=cfg["wd"], mix_cfg=cfg["mix_cfg"],
                                 use_ema=cfg.get("use_ema", False), ema_decay=cfg.get("ema_decay", 0.999))
        summaries.append(summary)

    # comparison CSV and PNG
    csv_path = os.path.join(RESULTS_DIR, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "val_acc", "val_loss", "train_time_sec"])
        for s in summaries: w.writerow([s["model"], s["val_acc"], s["val_loss"], s["train_time_sec"]])

    # bar plot
    names = [s["model"] for s in summaries]
    accs = [s["val_acc"] for s in summaries]
    losses = [s["val_loss"] for s in summaries]
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1); plt.bar(names, accs); plt.ylim(0,1); plt.title("Val Accuracy")
    plt.subplot(1,2,2); plt.bar(names, losses); plt.title("Val Loss")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "comparison_summary.png")); plt.close()

    print("All done. Results in", RESULTS_DIR)

if __name__ == "__main__":
    main()
