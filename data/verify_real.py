"""
verify_real.py
15-minute interpolation verification against real GOES-19 ground truth.
Input: T0 (18:01 UTC) and T1 (18:31 UTC)
Predicts: 18:16 UTC (15-min midpoint)
Compares: against real GOES-19 frame at 18:16 UTC
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os, sys
import numpy as np
import netCDF4 as nc
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rife_tir import RIFE_TIR
from data.dataset import normalise, denormalise
from data.calibrate import radiance_to_bt
from data.coregister import get_lat_lon, reproject
from validate import compute_ssim_metric, compute_fsim

# ── Configuration ─────────────────────────────────────────────────────────────
SAVE_DIR   = r"D:\satellite-interp\data\verification_test_day178_18utc"
OUTPUT_DIR = r"D:\satellite-interp\outputs\verification"
CKPT_PATH  = r"D:\satellite-interp\outputs\checkpoints\best.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BUCKET     = "noaa-goes19"

TARGET_FILES = {
    "T0":  "ABI-L1b-RadC/2025/178/18/OR_ABI-L1b-RadC-M6C13_G19_s20251781801173_e20251781803558_c20251781804043.nc",
    "GT":  "ABI-L1b-RadC/2025/178/18/OR_ABI-L1b-RadC-M6C13_G19_s20251781816173_e20251781818559_c20251781819051.nc",
    "T1":  "ABI-L1b-RadC/2025/178/18/OR_ABI-L1b-RadC-M6C13_G19_s20251781831173_e20251781833558_c20251781834046.nc",
}


# ── Download ──────────────────────────────────────────────────────────────────
def download_files():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    os.makedirs(SAVE_DIR, exist_ok=True)
    paths = {}
    print("Downloading frames...")
    for label, key in TARGET_FILES.items():
        filename  = os.path.basename(key)
        save_path = os.path.join(SAVE_DIR, filename)
        paths[label] = save_path
        if os.path.exists(save_path):
            print(f"  {label} — exists: {filename}")
            continue
        print(f"  {label} — downloading...")
        s3.download_file(BUCKET, key, save_path)
        print(f"  {label} — done")
    return paths


# ── Calibrate + reproject ─────────────────────────────────────────────────────
def nc_to_grid(nc_path):
    with nc.Dataset(nc_path, "r") as ds:
        radiance = ds.variables["Rad"][:].astype(np.float32)
        fk1 = float(ds.variables["planck_fk1"][:])
        fk2 = float(ds.variables["planck_fk2"][:])
        bc1 = float(ds.variables["planck_bc1"][:])
        bc2 = float(ds.variables["planck_bc2"][:])
    bt  = radiance_to_bt(radiance, fk1, fk2, bc1, bc2)
    bt  = np.where((bt >= 180) & (bt <= 320), bt, np.nan)
    lat, lon = get_lat_lon(nc_path)
    return reproject(bt, lat, lon)


# ── Model inference ───────────────────────────────────────────────────────────
def interpolate_frame(model, arr_t0, arr_t1, t_value):
    PATCH, STRIDE = 128, 64
    norm_t0 = np.where(np.isnan(arr_t0), 0.5, normalise(arr_t0))
    norm_t1 = np.where(np.isnan(arr_t1), 0.5, normalise(arr_t1))
    H, W    = norm_t0.shape
    output  = np.zeros((H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    window  = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)

    p0, p1, pos = [], [], []
    for r in range(0, H - PATCH + 1, STRIDE):
        for c in range(0, W - PATCH + 1, STRIDE):
            p0.append(norm_t0[r:r+PATCH, c:c+PATCH])
            p1.append(norm_t1[r:r+PATCH, c:c+PATCH])
            pos.append((r, c))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(p0), 16):
            b0 = torch.tensor(np.stack(p0[i:i+16])).unsqueeze(1).to(DEVICE)
            b1 = torch.tensor(np.stack(p1[i:i+16])).unsqueeze(1).to(DEVICE)
            pred, _, _ = model(b0, b1, t=t_value)
            pred_np = pred.squeeze(1).cpu().numpy()
            for j, (r, c) in enumerate(pos[i:i+16]):
                output [r:r+PATCH, c:c+PATCH] += pred_np[j] * window
                weights[r:r+PATCH, c:c+PATCH] += window

    weights = np.where(weights < 1e-6, 1.0, weights)
    bt_out  = denormalise(output / weights)
    return np.where(np.isnan(arr_t0) & np.isnan(arr_t1), np.nan, bt_out).astype(np.float32)


# ── Metrics ───────────────────────────────────────────────────────────────────
def get_metrics(pred, gt):
    valid  = (~np.isnan(pred)) & (~np.isnan(gt))
    pred_v = pred[valid]
    gt_v   = gt[valid]
    pred_t = torch.tensor(normalise(pred_v)).unsqueeze(0).unsqueeze(0)
    gt_t   = torch.tensor(normalise(gt_v  )).unsqueeze(0).unsqueeze(0)
    mse    = float(np.mean((pred_v - gt_v) ** 2))
    psnr   = float(10 * np.log10(140.0**2 / mse)) if mse > 1e-10 else 100.0
    return {
        "mse":  round(mse, 4),
        "psnr": round(psnr, 4),
        "ssim": round(compute_ssim_metric(pred_t, gt_t), 4),
        "fsim": round(compute_fsim(pred_t, gt_t), 4),
    }


# ── Visualise ─────────────────────────────────────────────────────────────────
def visualise(t0, gt, pred, t1, save_path):
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)
    vmin, vmax, cmap = 200, 312, "gray_r"

    frames = [
        (t0,   "T0 — 18:01 UTC\n(real input)"),
        (gt,   "18:16 UTC\n(REAL ground truth)"),
        (pred, "18:16 UTC\n(MODEL predicted t=0.5)"),
        (t1,   "T1 — 18:31 UTC\n(real input)"),
    ]

    for col, (arr, title) in enumerate(frames):
        ax = fig.add_subplot(gs[col])
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        color = "navy" if "MODEL" in title else "black"
        ax.set_title(title, fontsize=10, fontweight="bold", color=color)
        ax.axis("off")

    im = plt.matplotlib.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(im, ax=fig.get_axes(), shrink=0.8,
                 label="Brightness Temperature (K)", pad=0.01)

    plt.suptitle(
        "15-min Interpolation: GOES-19 Real vs RIFE-TIR Predicted\n"
        "Input: 18:01 & 18:31 UTC  →  Predicted: 18:16 UTC",
        fontsize=12, fontweight="bold"
    )
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Image saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("15-min Interpolation — Ground Truth Verification")
    print("T0=18:01  GT=18:16  T1=18:31  (GOES-19 Day 178)")
    print("=" * 60)

    # Download
    paths = download_files()

    # Calibrate
    print("\nCalibrating frames...")
    grids = {}
    for label, path in paths.items():
        grids[label] = nc_to_grid(path)
        print(f"  {label}: {np.nanmin(grids[label]):.1f} – {np.nanmax(grids[label]):.1f} K")

    # Load model
    print(f"\nLoading model...")
    device = torch.device(DEVICE)
    model  = RIFE_TIR().to(device)
    ckpt   = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded epoch {ckpt['epoch']}")

    # Predict 15-min frame
    print("\nPredicting 15-min frame (t=0.5)...")
    pred = interpolate_frame(model, grids["T0"], grids["T1"], t_value=0.5)

    # Linear blend baseline
    blend = denormalise(
        0.5 * normalise(np.where(np.isnan(grids["T0"]), 0, grids["T0"])) +
        0.5 * normalise(np.where(np.isnan(grids["T1"]), 0, grids["T1"]))
    )

    # Metrics
    print("\nComputing metrics...")
    m_model = get_metrics(pred,  grids["GT"])
    m_blend = get_metrics(blend, grids["GT"])

    # Report
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT — 15-min Interpolation")
    print("Predicted 18:16 UTC  vs  Real GOES-19 18:16 UTC")
    print("=" * 60)
    print(f"\n  {'Metric':<8} {'Linear Blend':>14} {'RIFE-TIR':>12} {'Diff':>8} {'Better?':>8}")
    print(f"  {'-'*54}")
    for metric in ["ssim", "psnr", "mse", "fsim"]:
        b = m_blend[metric]
        m = m_model[metric]
        if metric == "mse":
            better = "✓" if m < b else "✗"
            diff   = f"{((b-m)/b*100):+.1f}%"
        else:
            better = "✓" if m > b else "✗"
            diff   = f"{((m-b)/b*100):+.1f}%"
        print(f"  {metric.upper():<8} {b:>14.4f} {m:>12.4f} {diff:>8} {better}")

    # Visualise
    print()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    viz_path = os.path.join(OUTPUT_DIR, "15min_verification.png")
    visualise(grids["T0"], grids["GT"], pred, grids["T1"], viz_path)

    print("\nDone.")