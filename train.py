"""
train.py
Training loop for RIFE-TIR satellite frame interpolation model.
Run locally for testing (CPU), push to Kaggle for full GPU training.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.rife_tir import RIFE_TIR, physics_aware_loss
from data.dataset import SatelliteTripletDataset, load_index

# ── Console for pretty output ─────────────────────────────────────────────────
console = Console()

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Paths
    "index_file":   r"D:\satellite-interp\data\triplet_index.json",
    "ckpt_dir":     r"D:\satellite-interp\outputs\checkpoints",
    "log_file":     r"D:\satellite-interp\outputs\train_log.json",

    # Fine-tune settings — lower LR, fewer epochs
    "epochs":       40,          # changed from 30 → 40
    "batch_size":   4,
    "lr":           1e-5,        # changed from 1e-4 → 1e-5 (fine-tune LR)
    "weight_decay": 1e-4,
    "grad_clip":    1.0,
    "num_workers":  0,
    "val_every":    1,
    "save_every":   5,

    # Early stopping
    "patience":     10,

    # Device
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

 
# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, best_ssim, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "best_ssim":  best_ssim,
    }, path)

def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["best_ssim"]


# ── SSIM metric (for validation) ─────────────────────────────────────────────
def compute_ssim(pred, target):
    C1, C2 = 0.01**2, 0.03**2
    import torch.nn.functional as F
    mu_p  = F.avg_pool2d(pred,   3, 1, 1)
    mu_t  = F.avg_pool2d(target, 3, 1, 1)
    mu_pp = F.avg_pool2d(pred*pred,     3, 1, 1)
    mu_tt = F.avg_pool2d(target*target, 3, 1, 1)
    mu_pt = F.avg_pool2d(pred*target,   3, 1, 1)
    sig_p  = mu_pp - mu_p*mu_p
    sig_t  = mu_tt - mu_t*mu_t
    sig_pt = mu_pt - mu_p*mu_t
    ssim = ((2*mu_p*mu_t+C1)*(2*sig_pt+C2)) / \
           ((mu_p**2+mu_t**2+C1)*(sig_p+sig_t+C2))
    return ssim.mean().item()


# ── Training step ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    components_sum = {
        "pixel": 0, "ssim": 0, "therm": 0,
        "mass": 0, "smooth": 0, "div": 0
    }

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d} [train]", leave=False)
    for batch in pbar:
        t0, t1, gt, t_param = [x.to(device) for x in batch]

        optimizer.zero_grad()
        pred, flow, mask = model(t0, t1, t=t_param.mean().item())
        loss, comps      = physics_aware_loss(pred, flow, t0, t1, gt, t_param)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()

        total_loss += loss.item()
        for k in components_sum:
            components_sum[k] += comps[k]

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    avg_loss = total_loss / n
    avg_comps = {k: v/n for k, v in components_sum.items()}
    return avg_loss, avg_comps


# ── Validation step ───────────────────────────────────────────────────────────
def validate(model, loader, device, epoch):
    model.eval()
    total_loss = 0
    total_ssim = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"  Epoch {epoch:03d} [val]  ", leave=False)
        for batch in pbar:
            t0, t1, gt, t_param = [x.to(device) for x in batch]
            pred, flow, mask    = model(t0, t1, t=t_param.mean().item())
            loss, _             = physics_aware_loss(pred, flow, t0, t1, gt, t_param)
            ssim                = compute_ssim(pred, gt)

            total_loss += loss.item()
            total_ssim += ssim
            pbar.set_postfix(ssim=f"{ssim:.4f}")

    n = len(loader)
    return total_loss / n, total_ssim / n


# ── Main training loop ────────────────────────────────────────────────────────
def train():
    console.rule("[bold blue]RIFE-TIR Training")
    console.print(f"Device     : [green]{CONFIG['device']}")
    console.print(f"Epochs     : {CONFIG['epochs']}")
    console.print(f"Batch size : {CONFIG['batch_size']}")
    console.print(f"LR         : {CONFIG['lr']}")
    console.print()

    # ── Data ──────────────────────────────────────────────────────────────────
    console.print("[bold]Loading dataset...")
    triplets = load_index()
    split    = int(len(triplets) * 0.8)
    train_t  = triplets[:split]
    val_t    = triplets[split:]

    train_ds = SatelliteTripletDataset(train_t, augment=True)
    val_ds   = SatelliteTripletDataset(val_t,   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["device"] == "cuda"
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )

    console.print(f"Train patches : {len(train_ds)}")
    console.print(f"Val patches   : {len(val_ds)}")
    console.print()

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device(CONFIG["device"])
    model  = RIFE_TIR().to(device)
    console.print(f"[bold]Model params  : {model.total_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],
        eta_min=1e-6
    )

    # ── Resume from checkpoint if exists ──────────────────────────────────────
    best_ckpt  = os.path.join(CONFIG["ckpt_dir"], "best.pth")
    start_epoch = 1
    best_ssim   = 0.0

    last_ckpt = os.path.join(CONFIG["ckpt_dir"], "last.pth")
    if os.path.exists(last_ckpt):
        console.print(f"[yellow]Resuming from {last_ckpt}")
        start_epoch, best_ssim = load_checkpoint(model, optimizer, last_ckpt)
        start_epoch += 1

    # ── Training log ──────────────────────────────────────────────────────────
    log = []
    patience_counter = 0

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        t_start = time.time()

        # Train
        train_loss, train_comps = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # Validate
        val_loss, val_ssim = validate(model, val_loader, device, epoch)

        scheduler.step()
        elapsed = time.time() - t_start

        # ── Logging ───────────────────────────────────────────────────────────
        log_entry = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "val_ssim":   round(val_ssim,   6),
            "lr":         round(scheduler.get_last_lr()[0], 8),
            "time_s":     round(elapsed, 1),
            **{f"train_{k}": round(v, 6) for k, v in train_comps.items()}
        }
        log.append(log_entry)

        # Save log
        os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)
        with open(CONFIG["log_file"], "w") as f:
            json.dump(log, f, indent=2)

        # ── Print epoch summary ───────────────────────────────────────────────
        improved = "⬆ best" if val_ssim > best_ssim else ""
        console.print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"SSIM={val_ssim:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.0f}s {improved}"
        )

        # ── Save checkpoints ──────────────────────────────────────────────────
        # Always save last
        save_checkpoint(model, optimizer, epoch, best_ssim, last_ckpt)

        # Save best
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_ssim, best_ckpt)
            console.print(f"  [green]New best SSIM: {best_ssim:.4f} — saved best.pth")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % CONFIG["save_every"] == 0:
            path = os.path.join(CONFIG["ckpt_dir"], f"epoch_{epoch:03d}.pth")
            save_checkpoint(model, optimizer, epoch, best_ssim, path)

        # Early stopping
        if patience_counter >= CONFIG["patience"]:
            console.print(f"\n[yellow]Early stopping at epoch {epoch} "
                          f"(no improvement for {CONFIG['patience']} epochs)")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    console.rule("[bold green]Training Complete")
    console.print(f"Best SSIM  : [green]{best_ssim:.4f}")
    console.print(f"Checkpoint : {best_ckpt}")
    console.print(f"Log        : {CONFIG['log_file']}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()