"""
verify_real.py
Downloads 5 specific GOES-19 frames:
  T0    = 06:01 UTC  (input)
  GT10  = 06:11 UTC  (real 10-min — ground truth)
  GT20  = 06:21 UTC  (real 20-min — ground truth)
  T1    = 06:31 UTC  (input)
  GT30  = 06:41 UTC  (bonus check)

Model interpolates T0→T1 to get predicted 10-min and 20-min frames.
We then compare predicted vs real and report SSIM, PSNR, MSE, FSIM.
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
from data.coregister import get_lat_lon, reproject, TARGET_LATS, TARGET_LONS, GRID_H, GRID_W
from validate import compute_mse, compute_psnr, compute_ssim_metric, compute_fsim

# ── Configuration ─────────────────────────────────────────────────────────────
SAVE_DIR   = r"D:\satellite-interp\data\verification_test"
OUTPUT_DIR = r"D:\satellite-interp\outputs\verification"
CKPT_PATH  = r"D:\satellite-interp\outputs\checkpoints\best.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BUCKET     = "noaa-goes19"

# Exact files we want — day 177, hour 06
TARGET_FILES = {
    "T0":   "ABI-L1b-RadC/2025/177/06/OR_ABI-L1b-RadC-M6C13_G19_s20251770601172_e20251770603556_c20251770604038.nc",
    "GT10": "ABI-L1b-RadC/2025/177/06/OR_ABI-L1b-RadC-M6C13_G19_s20251770611172_e20251770613557_c20251770614009.nc",
    "GT20": "ABI-L1b-RadC/2025/177/06/OR_ABI-L1b-RadC-M6C13_G19_s20251770621172_e20251770623557_c20251770624037.nc",
    "T1":   "ABI-L1b-RadC/2025/177/06/OR_ABI-L1b-RadC-M6C13_G19_s20251770631172_e20251770633557_c20251770634038.nc",
}


# ── Step 1: Download ──────────────────────────────────────────────────────────
def download_files():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    os.makedirs(SAVE_DIR, exist_ok=True)
    paths = {}

    print("Downloading verification frames...")
    for label, key in TARGET_FILES.items():
        filename  = os.path.basename(key)
        save_path = os.path.join(SAVE_DIR, filename)
        paths[label] = save_path

        if os.path.exists(save_path):
            print(f"  {label} — already exists: {filename}")
            continue

        print(f"  {label} — downloading: {filename}")
        s3.download_file(BUCKET, key, save_path)
        print(f"  {label} — done")

    return paths


# ── Step 2: Calibrate + reproject one file ────────────────────────────────────
def nc_to_grid(nc_path):
    """Full pipeline: .nc → BT → reprojected grid array."""
    with nc.Dataset(nc_path, "r") as ds:
        rad_var  = ds.variables["Rad"]
        radiance = rad_var[:].astype(np.float32)
        fk1 = float(ds.variables["planck_fk1"][:])
        fk2 = float(ds.variables["planck_fk2"][:])
        bc1 = float(ds.variables["planck_bc1"][:])
        bc2 = float(ds.variables["planck_bc2"][:])

    bt = radiance_to_bt(radiance, fk1, fk2, bc1, bc2)
    bt = np.where((bt >= 180) & (bt <= 320), bt, np.nan)

    lat, lon = get_lat_lon(nc_path)
    grid     = reproject(bt, lat, lon)
    return grid


# ── Step 3: Model inference on full frame ─────────────────────────────────────
def interpolate_full_frame(model, arr_t0, arr_t1, t_value):
    """Run model on full frame using patch tiling."""
    PATCH = 128
    STRIDE = 64

    norm_t0 = np.where(np.isnan(arr_t0), 0.5, normalise(arr_t0))
    norm_t1 = np.where(np.isnan(arr_t1), 0.5, normalise(arr_t1))

    H, W = norm_t0.shape
    output  = np.zeros((H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)

    win_1d = np.hanning(PATCH).astype(np.float32)
    window = np.outer(win_1d, win_1d)

    patches_t0, patches_t1, positions = [], [], []
    for r in range(0, H - PATCH + 1, STRIDE):
        for c in range(0, W - PATCH + 1, STRIDE):
            patches_t0.append(norm_t0[r:r+PATCH, c:c+PATCH])
            patches_t1.append(norm_t1[r:r+PATCH, c:c+PATCH])
            positions.append((r, c))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(patches_t0), 16):
            bt0 = torch.tensor(np.stack(patches_t0[i:i+16])).unsqueeze(1).to(DEVICE)
            bt1 = torch.tensor(np.stack(patches_t1[i:i+16])).unsqueeze(1).to(DEVICE)
            pred, _, _ = model(bt0, bt1, t=t_value)
            pred_np = pred.squeeze(1).cpu().numpy()

            for j, (r, c) in enumerate(positions[i:i+16]):
                output [r:r+PATCH, c:c+PATCH] += pred_np[j] * window
                weights[r:r+PATCH, c:c+PATCH] += window

    weights = np.where(weights < 1e-6, 1.0, weights)
    stitched = output / weights
    bt_out   = denormalise(stitched)
    nan_mask = np.isnan(arr_t0) & np.isnan(arr_t1)
    return np.where(nan_mask, np.nan, bt_out).astype(np.float32)


# ── Step 4: Compute metrics ───────────────────────────────────────────────────
def get_metrics(pred_arr, gt_arr):
    """Compute all 4 metrics between two numpy arrays."""
    # Use only pixels valid in both
    valid = (~np.isnan(pred_arr)) & (~np.isnan(gt_arr))
    if valid.sum() < 100:
        return None

    pred_v = pred_arr[valid]
    gt_v   = gt_arr[valid]

    # Convert to tensors for metric functions
    pred_t = torch.tensor(normalise(pred_v)).unsqueeze(0).unsqueeze(0)
    gt_t   = torch.tensor(normalise(gt_v  )).unsqueeze(0).unsqueeze(0)

    mse  = float(np.mean((pred_v - gt_v) ** 2))
    psnr = float(10 * np.log10(140.0**2 / mse)) if mse > 1e-10 else 100.0
    ssim = compute_ssim_metric(pred_t, gt_t)
    fsim = compute_fsim(pred_t, gt_t)

    return {"mse": round(mse, 4), "psnr": round(psnr, 4),
            "ssim": round(ssim, 4), "fsim": round(fsim, 4)}


# ── Step 5: Visualise ─────────────────────────────────────────────────────────
def visualise(t0, gt10, gt20, t1, pred10, pred20, save_path):
    fig = plt.figure(figsize=(18, 8))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.05)

    vmin, vmax = 200, 310
    cmap = "gray_r"

    frames_top = [
        (t0,    "T0 — 06:01 UTC\n(real input)"),
        (gt10,  "06:11 UTC\n(REAL ground truth)"),
        (gt20,  "06:21 UTC\n(REAL ground truth)"),
        (t1,    "T1 — 06:31 UTC\n(real input)"),
    ]
    frames_bot = [
        (None,   ""),
        (pred10, "06:11 UTC\n(MODEL predicted)"),
        (pred20, "06:21 UTC\n(MODEL predicted)"),
        (None,   ""),
    ]

    for col, (arr, title) in enumerate(frames_top):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    for col, (arr, title) in enumerate(frames_bot):
        ax = fig.add_subplot(gs[1, col])
        if arr is not None:
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(title, fontsize=10, fontweight="bold", color="navy")
        ax.axis("off")

    # Arrows showing what's real vs predicted
    fig.text(0.5, 0.52,
             "↑ Real GOES-19 frames       ↓ Model interpolated frames",
             ha="center", fontsize=11, color="darkgreen", fontweight="bold")

    plt.suptitle(
        "Ground Truth Verification: GOES-19 Real vs RIFE-TIR Predicted\n"
        "Input: 00-min & 30-min frames only  →  Output: 10-min & 20-min frames",
        fontsize=12, fontweight="bold"
    )

    im = plt.matplotlib.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    fig.colorbar(im, ax=fig.get_axes(), shrink=0.6,
                 label="Brightness Temperature (K)", pad=0.01)

    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("GOES-19 Ground Truth Verification Test")
    print("Real 10-min & 20-min frames vs Model predictions")
    print("=" * 60)

    # Step 1 — Download
    paths = download_files()

    # Step 2 — Calibrate + reproject all 4 frames
    print("\nCalibrating and reprojecting frames...")
    grids = {}
    for label, nc_path in paths.items():
        print(f"  Processing {label}...")
        grids[label] = nc_to_grid(nc_path)
        print(f"  {label} BT range: "
              f"{np.nanmin(grids[label]):.1f} – {np.nanmax(grids[label]):.1f} K")

    # Step 3 — Load model
    print(f"\nLoading model from {CKPT_PATH}...")
    device = torch.device(DEVICE)
    model  = RIFE_TIR().to(device)
    ckpt   = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Model loaded (epoch {ckpt['epoch']})")

    # Step 4 — Interpolate
    print("\nRunning interpolation...")
    print("  Predicting t=0.333 (10-min frame)...")
    pred10 = interpolate_full_frame(model, grids["T0"], grids["T1"], t_value=0.333)

    print("  Predicting t=0.667 (20-min frame)...")
    pred20 = interpolate_full_frame(model, grids["T0"], grids["T1"], t_value=0.667)

    # Step 5 — Compute metrics
    print("\nComputing metrics vs real ground truth frames...")
    m10 = get_metrics(pred10, grids["GT10"])
    m20 = get_metrics(pred20, grids["GT20"])

    # Also compute linear blend baseline
    blend10 = denormalise(0.333 * normalise(np.where(np.isnan(grids["T0"]), 0, grids["T0"])) +
                          0.667 * normalise(np.where(np.isnan(grids["T1"]), 0, grids["T1"])))
    blend20 = denormalise(0.667 * normalise(np.where(np.isnan(grids["T0"]), 0, grids["T0"])) +
                          0.333 * normalise(np.where(np.isnan(grids["T1"]), 0, grids["T1"])))
    b10 = get_metrics(blend10, grids["GT10"])
    b20 = get_metrics(blend20, grids["GT20"])

    # Step 6 — Print report
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT — vs Real GOES-19 Ground Truth")
    print("=" * 60)

    for label, m_model, m_base in [("10-min frame", m10, b10),
                                    ("20-min frame", m20, b20)]:
        print(f"\n{label}:")
        print(f"  {'Metric':<8} {'Linear Blend':>14} {'RIFE-TIR':>12} {'Better?':>10}")
        print(f"  {'-'*48}")
        for metric in ["ssim", "psnr", "mse", "fsim"]:
            b = m_base[metric]
            m = m_model[metric]
            if metric == "mse":
                better = "✓" if m < b else "✗"
                diff   = f"{((b-m)/b*100):+.1f}%"
            else:
                better = "✓" if m > b else "✗"
                diff   = f"{((m-b)/b*100):+.1f}%"
            print(f"  {metric.upper():<8} {b:>14.4f} {m:>12.4f} {diff:>8} {better}")

    # Step 7 — Visualise
    print("\nGenerating comparison image...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    viz_path = os.path.join(OUTPUT_DIR, "ground_truth_verification.png")
    visualise(grids["T0"], grids["GT10"], grids["GT20"],
              grids["T1"], pred10, pred20, viz_path)

    print("\n" + "=" * 60)
    print("Verification complete.")
    print(f"Image saved to: {viz_path}")
    print("=" * 60)