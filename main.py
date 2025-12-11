import os
import json
import time
import csv
import logging
from datetime import datetime
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------
# CONFIG
# -------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4
LOG_ROOT = "results"
USE_TORCH_COMPILE = True
USE_AMP = True

os.makedirs(LOG_ROOT, exist_ok=True)

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------
# Logging helper
# -------------------------
def get_logger(name, folder):
    os.makedirs(folder, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(folder, f"{name}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    if not logger.handlers:
        logger.addHandler(fh)
    return logger

# -------------------------
# Data
# -------------------------
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
classes = train_ds.classes

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# -------------------------
# Models
# -------------------------
# ResNet-18 baseline (from torchvision)
def build_resnet18(num_classes=10):
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # adapt for 32x32
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# Tiny ViT baseline
class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0, num_classes=10):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim*mlp_ratio), activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: B,3,32,32
        B = x.shape[0]
        x = self.patch_embed(x)  # B,embed,H',W'
        x = x.flatten(2).transpose(1,2)  # B, N, C
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)  # B, N+1, C
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        cls_out = x[:,0]
        return self.head(cls_out)

# Hybrid model (optimized) - reusing concepts from previous optimized script
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
    def __init__(self, C, patch_size=(4,4), heads=2):
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
        x_p = x.unfold(2, ph, ph).unfold(3, pw, pw)  # B,C,Hn,Wn,ph,pw
        x_p = x_p.contiguous().view(B, C, Hn, Wn, ph*pw).mean(-1)  # B,C,Hn,Wn
        x_p = x_p.permute(0,2,3,1).reshape(B, Hn*Wn, C)
        qkv = self.qkv(x_p).reshape(B, Hn*Wn, 3, self.heads, self.hd).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: B,heads,N,hd
        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        attn = attn.softmax(-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.permute(0,2,1,3).reshape(B, Hn, Wn, C).permute(0,3,1,2)
        out = F.interpolate(out, size=(H,W), mode="nearest")
        return out

class HybridBlockOptimized(nn.Module):
    def __init__(self, C, heads=1, patch_size=(4,4), gate_reduction=4, norm_layer=nn.BatchNorm2d):
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
    def __init__(self, num_classes=10, channels=(48,96,192), heads=(1,2,2), patch_sizes=((8,8),(4,4),(2,2))):
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
        return (logits, gate_list) if collect_gates else logits

# -------------------------
# Training utilities
# -------------------------
def maybe_compile(model):
    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            return model
        except Exception:
            return model
    return model

def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x,y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
    return total_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, collect_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x,y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
        if collect_preds:
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    if collect_preds:
        return total_loss/total, correct/total, torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()
    return total_loss/total, correct/total, None, None

# -------------------------
# IO helpers
# -------------------------
def ensure_dirs(base, model_name):
    root = os.path.join(LOG_ROOT, model_name)
    ckpt = os.path.join(root, "checkpoints")
    plots = os.path.join(root, "plots")
    os.makedirs(root, exist_ok=True)
    os.makedirs(ckpt, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return root, ckpt, plots

def save_checkpoint(model, optimizer, epoch, history, ckpt_path):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history}
    torch.save(payload, ckpt_path)

def load_checkpoint(model, optimizer, ckpt_path):
    if not os.path.exists(ckpt_path):
        return None
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    if optimizer is not None and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    return ck

def save_history_json(history, path):
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

def plot_history(history, out_png, title_prefix=""):
    epochs = range(1, len(history["train_loss"])+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], '-o', label='train_loss')
    plt.plot(epochs, history["val_loss"], '--o', label='val_loss')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{title_prefix} Loss"); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], '-o', label='train_acc')
    plt.plot(epochs, history["val_acc"], '--o', label='val_acc')
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{title_prefix} Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png); plt.close()

def plot_confusion(cm, labels, out_png, title="Confusion Matrix"):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png); plt.close()

# -------------------------
# Experiment runner
# -------------------------
def run_experiment(model_builder, model_name, epochs=EPOCHS, lr=LR):
    root, ckpt_dir, plots_dir = ensure_dirs(LOG_ROOT, model_name)
    logger = get_logger(model_name, root)
    logger.info(f"Starting experiment: {model_name} on device {DEVICE}")

    model = model_builder().to(DEVICE)
    model = maybe_compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE.startswith("cuda")) else None

    ckpt_path = os.path.join(ckpt_dir, f"{model_name}.pt")
    history, start_epoch = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}, 0

    loaded = load_checkpoint(model, optimizer, ckpt_path)
    if loaded:
        history = loaded.get("history", history) if isinstance(loaded, dict) else history
        start_epoch = loaded.get("epoch", 0)
        logger.info(f"Resuming from epoch {start_epoch}")

    t0 = time.time()
    for epoch in range(start_epoch + 1, epochs + 1):
        ep_start = time.time()
        print(f"[{model_name}] Epoch {epoch}/{epochs} ...")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler=scaler)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, collect_preds=False)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - ep_start
        eta = (epochs - epoch) * epoch_time
        logger.info(f"Epoch {epoch}: tr_loss={tr_loss:.4f}, tr_acc={tr_acc*100:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%, time={epoch_time:.2f}s")
        print(f"[{model_name}] Epoch {epoch} done. val_acc={val_acc*100:.2f}%, epoch_time={epoch_time:.2f}s, ETA={eta/60:.2f}min")

        # save checkpoint + history
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history}
        torch.save(payload, ckpt_path)
        save_history_json(history, os.path.join(root, f"{model_name}_history.json"))

    total_time = time.time() - t0
    print(f"[{model_name}] Training finished in {total_time/60:.2f} minutes")
    logger.info(f"Training finished. total_time={total_time:.2f}s")

    # final evaluation with predictions
    val_loss, val_acc, preds, labels_arr = evaluate(model, test_loader, criterion, collect_preds=True)
    cm = confusion_matrix(labels_arr, preds)
    plot_history(history, os.path.join(plots_dir, f"{model_name}_history.png"), title_prefix=model_name)
    plot_confusion(cm, classes, os.path.join(plots_dir, f"{model_name}_confmat.png"), title=f"{model_name} Confusion Matrix")

    # classification report to file
    crep = classification_report(labels_arr, preds, target_names=classes, digits=4)
    with open(os.path.join(root, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(crep)

    # save final metrics and history
    final = {
        "model": model_name,
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "train_time_sec": total_time,
        "history": history
    }
    with open(os.path.join(root, f"{model_name}_summary.json"), "w") as f:
        json.dump(final, f, indent=2)

    return final

# -------------------------
# Main orchestration
# -------------------------
def main():
    experiments = OrderedDict([
        ("ResNet18", lambda: build_resnet18(num_classes=10)),
        ("TinyViT", lambda: TinyViT(img_size=32, patch_size=4, embed_dim=128, depth=4, num_heads=4, num_classes=10)),
        ("GATENetOpt", lambda: GATENetOptimized(num_classes=10))
    ])

    summaries = []
    for name, builder in experiments.items():
        print(f"\n=== Running: {name} ===")
        summary = run_experiment(builder, name, epochs=EPOCHS, lr=LR)
        summaries.append(summary)

    # write comparison CSV and PNG
    csv_path = os.path.join(LOG_ROOT, "comparison.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "val_acc", "val_loss", "train_time_sec"])
        for s in summaries:
            writer.writerow([s["model"], s["val_acc"], s["val_loss"], s["train_time_sec"]])

    # Plot comparison bar chart
    names = [s["model"] for s in summaries]
    accs = [s["val_acc"] for s in summaries]
    losses = [s["val_loss"] for s in summaries]

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.bar(names, accs)
    plt.ylabel("Val Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.ylim(0,1)
    plt.subplot(1,2,2)
    plt.bar(names, losses)
    plt.ylabel("Val Loss")
    plt.title("Validation Loss Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_ROOT, "comparison_summary.png"))
    plt.close()

    print("\nAll experiments finished. Results stored in 'results/' folder.")

if __name__ == "__main__":
    main()
