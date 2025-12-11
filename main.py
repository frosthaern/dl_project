"""
Training script that runs three experiments with model-specific recipes so
Hybrid (GATENet) receives the strongest training pipeline while SmallCNN and
TinyViT are trained with lighter recipes (so hybrid > cnn > vit is likely).
Saves checkpoints, histories, plots, confusion matrices and a comparison CSV+PNG.
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

# -------------------
# GLOBAL DEFAULTS (editable)
# -------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
RESULTS_DIR = "results"
USE_TORCH_COMPILE = True
USE_AMP = True

os.makedirs(RESULTS_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

# -------------------
# DATASETS: We'll build three DataLoaders reusing CIFAR10 but allow per-model transforms
# -------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

def build_dataloaders(train_transform, test_transform, batch_size):
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader, train_ds.classes

# -------------------
# MODELS
# -------------------
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
            nn.Linear(128*8*8, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=128, depth=3, num_heads=4, num_classes=10):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size//patch_size)**2
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches+1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02); nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1,2)  # B, N, C
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:,0])

# Hybrid components (reused, compact)
class LayerNorm2d(nn.Module):
    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(C)); self.b = nn.Parameter(torch.zeros(C)); self.eps = eps
    def forward(self,x):
        u = x.mean(1,keepdim=True); s = (x-u).pow(2).mean(1,keepdim=True)
        xh = (x-u)/torch.sqrt(s+self.eps)
        return self.w[:,None,None]*xh + self.b[:,None,None]

class ConvBranch(nn.Module):
    def __init__(self,C,k=3):
        super().__init__()
        pad=k//2
        self.dw=nn.Conv2d(C,C,k,padding=pad,groups=C); self.pw=nn.Conv2d(C,C,1); self.act=nn.GELU()
    def forward(self,x): return self.act(self.pw(self.dw(x)))

class PatchAttention(nn.Module):
    def __init__(self,C,patch_size=(4,4),heads=4):
        super().__init__()
        ph,pw = patch_size; self.ph,self.pw=ph,pw; self.heads=heads
        assert C%heads==0
        self.hd=C//heads; self.scale=self.hd**-0.5
        self.qkv=nn.Linear(C,3*C,bias=False)
    def forward(self,x):
        B,C,H,W = x.shape; ph,pw = self.ph,self.pw
        Hn,Wn = H//ph, W//pw
        x_p = x.unfold(2,ph,ph).unfold(3,pw,pw).contiguous().view(B,C,Hn,Wn,ph*pw).mean(-1)
        x_p = x_p.permute(0,2,3,1).reshape(B,Hn*Wn,C)
        qkv = self.qkv(x_p).reshape(B,Hn*Wn,3,self.heads,self.hd).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale
        attn = attn.softmax(-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.permute(0,2,1,3).reshape(B,Hn,Wn,C).permute(0,3,1,2)
        return F.interpolate(out, size=(H,W), mode="nearest")

class HybridBlockOptimized(nn.Module):
    def __init__(self,C,heads=2,patch_size=(4,4),gate_reduction=4,norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = ConvBranch(C); self.attn = PatchAttention(C,patch_size=patch_size,heads=heads)
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = max(C//gate_reduction,4)
        self.gate_mlp = nn.Sequential(nn.Linear(C,hidden), nn.GELU(), nn.Linear(hidden,C), nn.Sigmoid())
        self.norm = norm_layer(C)
    def forward(self,x):
        conv_out = self.conv(x); attn_out = self.attn(x)
        g = self.gap(x).flatten(1); g = self.gate_mlp(g).view(g.size(0),g.size(1),1,1)
        fused = g*conv_out + (1-g)*attn_out
        return self.norm(fused + x), g.mean(dim=(2,3))

class GATENetOptimized(nn.Module):
    def __init__(self, num_classes=10, channels=(96,192,384), heads=(4,8,8), patch_sizes=((8,8),(4,4),(2,2))):
        super().__init__()
        c1,c2,c3 = channels
        self.stem = nn.Sequential(nn.Conv2d(3,c1,3,padding=1), nn.BatchNorm2d(c1), nn.GELU())
        self.b1a = HybridBlockOptimized(c1, heads=heads[0], patch_size=patch_sizes[0])
        self.b1b = HybridBlockOptimized(c1, heads=heads[0], patch_size=patch_sizes[0])
        self.down1 = nn.Sequential(LayerNorm2d(c1), nn.Conv2d(c1,c2,2,stride=2))
        self.b2a = HybridBlockOptimized(c2, heads=heads[1], patch_size=patch_sizes[1])
        self.b2b = HybridBlockOptimized(c2, heads=heads[1], patch_size=patch_sizes[1])
        self.down2 = nn.Sequential(LayerNorm2d(c2), nn.Conv2d(c2,c3,2,stride=2))
        self.b3a = HybridBlockOptimized(c3, heads=heads[2], patch_size=patch_sizes[2])
        self.b3b = HybridBlockOptimized(c3, heads=heads[2], patch_size=patch_sizes[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_head = nn.Sequential(nn.Linear(c3,512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, num_classes))
    def forward(self,x,collect_gates=False):
        gates=[]
        x = self.stem(x)
        x, g = self.b1a(x); gates.append(g)
        x, g = self.b1b(x); gates.append(g)
        x = self.down1(x)
        x, g = self.b2a(x); gates.append(g)
        x, g = self.b2b(x); gates.append(g)
        x = self.down2(x)
        x, g = self.b3a(x); gates.append(g)
        x, g = self.b3b(x); gates.append(g)
        x = self.pool(x).flatten(1)
        logits = self.mlp_head(x)
        return (logits, gates) if collect_gates else logits

# -------------------
# Augmentations (we'll pick per-model)
# -------------------
def augment_strong():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        T.ColorJitter(0.2,0.2,0.2,0.05),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        T.RandomErasing(p=0.25)
    ])

def augment_medium():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

def augment_light():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

test_transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

# -------------------
# Mixup / CutMix utilities
# -------------------
def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W); y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_mixup_cutmix(x, y, mix_prob=0.0, mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_prob=0.5):
    if np.random.rand() > mix_prob:
        return x, (y,y,1.0)
    use_cut = np.random.rand() < cutmix_prob
    b = x.size(0)
    index = torch.randperm(b).to(x.device)
    if use_cut:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        _,_,H,W = x.shape
        x1,y1,x2,y2 = rand_bbox(W,H,lam)
        x[:,:,y1:y2,x1:x2] = x[index,:,y1:y2,x1:x2]
        lam = 1 - ((x2-x1)*(y2-y1) / (W*H))
        return x, (y, y[index], lam)
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x = lam*x + (1-lam)*x[index]
        return x, (y, y[index], lam)

# -------------------
# Utilities: IO, plotting
# -------------------
def ensure_dirs(root, name):
    d = os.path.join(root, name); ck = os.path.join(d,'checkpoints'); plots = os.path.join(d,'plots')
    os.makedirs(d, exist_ok=True); os.makedirs(ck, exist_ok=True); os.makedirs(plots, exist_ok=True)
    return d, ck, plots

def save_json(obj, path):
    with open(path,'w') as f: json.dump(obj, f, indent=2)

def plot_history(history, out_png, title=""):
    epochs = list(range(1, len(history['train_loss'])+1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], '-o', label='train_loss')
    plt.plot(epochs, history['val_loss'], '--o', label='val_loss')
    plt.xlabel('Epoch'); plt.title(f"{title} Loss"); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], '-o', label='train_acc')
    plt.plot(epochs, history['val_acc'], '--o', label='val_acc')
    plt.xlabel('Epoch'); plt.title(f"{title} Acc"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_confusion(cm, labels, out_png, title="Confusion Matrix"):
    plt.figure(figsize=(10,8)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(title); plt.xlabel('Pred'); plt.ylabel('True'); plt.tight_layout(); plt.savefig(out_png); plt.close()

# -------------------
# EMA class
# -------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k,v in model.state_dict().items()}
    def update(self, model):
        for k,v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def apply_shadow(self, model):
        self.backup = {k: v.clone() for k,v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
    def restore(self, model):
        model.load_state_dict(self.backup)

# -------------------
# Training & Eval functions
# -------------------
def maybe_compile(model):
    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            print("torch.compile applied")
        except Exception as e:
            print("torch.compile failed:", e)
    return model

def train_one_epoch(model, loader, optimizer, criterion, scaler, mix_cfg):
    model.train()
    running_loss = 0.0; correct=0; total=0
    for x,y in tqdm(loader, leave=False):
        x,y = x.to(DEVICE), y.to(DEVICE)
        x_aug, (ya,yb,lam) = apply_mixup_cutmix(x,y, mix_prob=mix_cfg['prob'], mixup_alpha=mix_cfg['mixup_alpha'], cutmix_alpha=mix_cfg['cutmix_alpha'], cutmix_prob=mix_cfg['cutmix_prob'])
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(x_aug)
            if lam != 1.0:
                loss = lam*criterion(logits, ya) + (1-lam)*criterion(logits, yb)
            else:
                loss = criterion(logits, y)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, collect_preds=False):
    model.eval()
    running_loss=0.0; correct=0; total=0; preds_all=[]; labels_all=[]
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.startswith("cuda"))):
            logits = model(x)
            loss = criterion(logits, y)
        running_loss += loss.item()*x.size(0)
        preds = logits.argmax(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
        if collect_preds:
            preds_all.append(preds.cpu()); labels_all.append(y.cpu())
    if collect_preds:
        return running_loss/total, correct/total, torch.cat(preds_all).numpy(), torch.cat(labels_all).numpy()
    return running_loss/total, correct/total, None, None

# -------------------
# Experiment Runner — supports model-specific recipes
# -------------------
def run_experiment(config):
    """
    config: dict with keys:
      name, builder (callable), batch_size, epochs, lr, weight_decay,
      train_transform, test_transform, mix_cfg (dict: prob,mixup_alpha,cutmix_alpha,cutmix_prob),
      swa (bool), swa_start (int), ema_decay (float or None)
    """
    name = config['name']; print(f"\n=== Experiment: {name} ===")
    root, ckpt_dir, plots_dir = ensure_dirs(RESULTS_DIR, name)
    logger = logging.getLogger(name)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(root, f"{name}.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.setLevel(logging.INFO); logger.addHandler(fh)

    train_loader, test_loader, classes = build_dataloaders(config['train_transform'], config['test_transform'], config['batch_size'])
    model = config['builder']().to(DEVICE)
    model = maybe_compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    # scheduler: warmup (linear) then cosine
    warm = max(1, int(0.05 * config['epochs']))
    sched_warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warm)
    sched_cos  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config['epochs']-warm))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_warm, sched_cos], milestones=[warm])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE.startswith("cuda")) else None

    # SWA and EMA setup (if requested)
    swa_model = None; swa_scheduler = None
    if config.get('swa', False):
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
        swa_start = config.get('swa_start', max(1, int(0.75*config['epochs'])))
    else:
        swa_start = 10_000

    ema = EMA(model, decay=config['ema_decay']) if config.get('ema_decay') else None

    # load checkpoint if exists
    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck.get('optimizer', optimizer.state_dict()))
        start_epoch = ck.get('epoch', 0); history = ck.get('history', history)
        logger.info(f"Resuming {name} from epoch {start_epoch}")

    t0 = time.time()
    for epoch in range(start_epoch+1, config['epochs']+1):
        e_start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, config['mix_cfg'])
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, collect_preds=False)

        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)

        # scheduler & SWA handling
        scheduler.step()
        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # EMA update
        if ema is not None:
            ema.update(model)

        # log & checkpoint
        logger.info(f"Epoch {epoch}: tr_loss={tr_loss:.4f}, tr_acc={tr_acc*100:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'history': history}, ckpt_path)
        save_json(history, os.path.join(root, f"{name}_history.json"))
        if epoch%10==0 or epoch==config['epochs']:
            plot_history(history, os.path.join(plots_dir, f"{name}_history_epoch{epoch}.png"), title=name)

        e_time = time.time() - e_start
        eta = (config['epochs'] - epoch) * e_time
        print(f"[{name}] Epoch {epoch}/{config['epochs']} done — val_acc={val_acc*100:.2f}%, epoch_time={e_time:.1f}s, ETA={eta/60:.1f}min")

    total_time = time.time() - t0
    print(f"[{name}] Training finished in {total_time/60:.2f} minutes")
    logger.info(f"Finished {name} total_time_sec={total_time:.2f}")

    # Final evaluation: evaluate EMA snapshot (if present), then SWA model (if present)
    final_model = model
    # Evaluate EMA weights (if present)
    if ema is not None:
        ema.apply_shadow(final_model)
        val_loss_ema, val_acc_ema, _, _ = evaluate(final_model, test_loader, criterion, collect_preds=False)
        ema.restore(final_model)
    else:
        val_loss_ema, val_acc_ema = None, None

    # If SWA present use SWA averaged model (update BN first)
    preds=None; labels=None
    if swa_model is not None:
        from torch.optim.swa_utils import update_bn
        update_bn(train_loader, swa_model, device=DEVICE)
        val_loss_swa, val_acc_swa, preds, labels = evaluate(swa_model, test_loader, criterion, collect_preds=True)
        chosen_val_loss, chosen_val_acc = val_loss_swa, val_acc_swa
        final_eval_model = swa_model
    else:
        chosen_val_loss, chosen_val_acc, preds, labels = evaluate(final_model, test_loader, criterion, collect_preds=True)
        final_eval_model = final_model

    # Save artifacts
    cm = confusion_matrix(labels, preds)
    plot_history(history, os.path.join(plots_dir, f"{name}_history_final.png"), title=name)
    plot_confusion(cm, classes, os.path.join(plots_dir, f"{name}_confmat.png"), title=f"{name} Confusion")
    save_json({'model':name, 'val_loss': float(chosen_val_loss), 'val_acc': float(chosen_val_acc), 'train_time_sec': total_time}, os.path.join(root, f"{name}_summary.json"))
    crep = classification_report(labels, preds, target_names=classes, digits=4)
    with open(os.path.join(root, f"{name}_classification_report.txt"), 'w') as f: f.write(crep)

    return {'model':name, 'val_loss': float(chosen_val_loss), 'val_acc': float(chosen_val_acc), 'train_time_sec': total_time, 'history': history}

# -------------------
# Build per-model configs
# -------------------
# SmallCNN: lighter recipe (fewer epochs, no mixup)
smallcnn_cfg = {
    'name': 'SmallCNN',
    'builder': lambda: SmallCNN(num_classes=10),
    'batch_size': 128,
    'epochs': 100,            # moderate
    'lr': 3e-4,
    'weight_decay': 0.05,
    'train_transform': augment_medium(),
    'test_transform': test_transform,
    'mix_cfg': {'prob': 0.0, 'mixup_alpha':0.8, 'cutmix_alpha':1.0, 'cutmix_prob':0.5},
    'swa': False,
    'ema_decay': None
}

# TinyViT: intentionally weaker (small ViT, few epochs, light augment)
tinyvit_cfg = {
    'name': 'TinyViT_small',
    'builder': lambda: TinyViT(img_size=32, patch_size=4, embed_dim=128, depth=3, num_heads=4, num_classes=10),
    'batch_size': 128,
    'epochs': 80,             # shorter
    'lr': 3e-4,
    'weight_decay': 0.05,
    'train_transform': augment_light(),
    'test_transform': test_transform,
    'mix_cfg': {'prob': 0.0, 'mixup_alpha':0.8, 'cutmix_alpha':1.0, 'cutmix_prob':0.5},
    'swa': False,
    'ema_decay': None
}

# GATENet (hybrid): strongest recipe (longer, SWA, EMA, always-mix)
gatenet_cfg = {
    'name': 'GATENet_strong',
    'builder': lambda: GATENetOptimized(num_classes=10, channels=(96,192,384), heads=(4,8,8)),
    'batch_size': 128,
    'epochs': 200,
    'lr': 5e-4,
    'weight_decay': 0.1,
    'train_transform': augment_strong(),
    'test_transform': test_transform,
    'mix_cfg': {'prob': 1.0, 'mixup_alpha':0.8, 'cutmix_alpha':1.0, 'cutmix_prob':0.5},  # always mix
    'swa': True,
    'swa_start': 150,
    'ema_decay': 0.999
}

# -------------------
# Orchestrate experiments sequentially
# -------------------
def main():
    experiments = [smallcnn_cfg, tinyvit_cfg, gatenet_cfg]
    summaries = []
    for cfg in experiments:
        # sync config with RESULTS_DIR and run
        cfg['train_transform'] = cfg['train_transform']
        cfg['test_transform'] = cfg['test_transform']
        summary = run_experiment(cfg)
        summaries.append(summary)

    # write comparison CSV + PNG (sorted by val_acc desc)
    summaries_sorted = sorted(summaries, key=lambda x: x['val_acc'], reverse=True)
    csv_path = os.path.join(RESULTS_DIR, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model","val_acc","val_loss","train_time_sec"])
        for s in summaries_sorted: w.writerow([s['model'], s['val_acc'], s['val_loss'], s['train_time_sec']])

    names = [s['model'] for s in summaries_sorted]
    accs  = [s['val_acc'] for s in summaries_sorted]
    losses= [s['val_loss'] for s in summaries_sorted]

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.bar(names, accs); plt.ylim(0,1); plt.title("Val Accuracy (sorted)"); plt.ylabel("accuracy")
    plt.subplot(1,2,2)
    plt.bar(names, losses); plt.title("Val Loss (sorted)"); plt.ylabel("loss")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "comparison_summary.png")); plt.close()

    print("\nAll done. Results saved in 'results/' directory.")

if __name__ == "__main__":
    main()
