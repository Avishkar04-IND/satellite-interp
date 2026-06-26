"""
inference.py
Accepts two consecutive GOES-19 grid files (simulating INSAT-3DS 30-min pairs)
and produces interpolated intermediate frames as .h5 files.

Usage:
    python inference.py --t0 frame1.npy --t1 frame2.npy --mode 3x
"""

import os
import sys
import argparse
import numpy as np
import torch
import h5py
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.rife_tir import RIFE_TIR
from data.dataset import normalise, denormalise

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT_PATH  = r"D:\satellite-interp\outputs\checkpoints\best.pth"
OUTPUT_DIR = r"D:\satellite-interp\outputs\interpolated"
PATCH_SIZE = 128
STRIDE     = 64
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# t values for each upscaling mode
MODES = {
    "2x":  [0.5],
    "3x":  [0.333, 0.667],
    "4x":  [0.25, 0.5, 0.75],
}


# ── Patch tiling helpers ──────────────────────────────────────────────────────

def extract_patches(arr, patch_size, stride):
    """Extract all patches and their positions from a 2D array."""
    H, W = arr.shape
    patches, positions = [], []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patches.append(arr[r:r+patch_size, c:c+patch_size])
            positions.append((r, c))
    return patches, positions, H, W


def stitch_patches(patches, positions, H, W, patch_size):
    """
    Reconstruct full frame from overlapping patches using
    Hanning window blending — eliminates seam artefacts.
    """
    output  = np.zeros((H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)

    # Hanning window for smooth blending at patch edges
    win_1d = np.hanning(patch_size).astype(np.float32)
    window = np.outer(win_1d, win_1d)

    for patch, (r, c) in zip(patches, positions):
        output [r:r+patch_size, c:c+patch_size] += patch  * window
        weights[r:r+patch_size, c:c+patch_size] += window

    # Avoid division by zero at borders
    weights = np.where(weights < 1e-6, 1.0, weights)
    return output / weights


# ── Core inference ────────────────────────────────────────────────────────────

def interpolate_pair(model, arr_t0, arr_t1, t_value, device):
    """
    Run model inference for a single t value.
    Processes full frame by tiling into patches then stitching.
    """
    # Normalise
    norm_t0 = normalise(arr_t0)
    norm_t1 = normalise(arr_t1)

    # Replace NaN with 0.5
    norm_t0 = np.where(np.isnan(norm_t0), 0.5, norm_t0)
    norm_t1 = np.where(np.isnan(norm_t1), 0.5, norm_t1)

    # Extract patches
    patches_t0, positions, H, W = extract_patches(norm_t0, PATCH_SIZE, STRIDE)
    patches_t1, _,         _, _ = extract_patches(norm_t1, PATCH_SIZE, STRIDE)

    pred_patches = []

    model.eval()
    with torch.no_grad():
        # Process in mini-batches of 16 patches
        batch_size = 16
        for i in range(0, len(patches_t0), batch_size):
            batch_t0 = torch.tensor(
                np.stack(patches_t0[i:i+batch_size]), dtype=torch.float32
            ).unsqueeze(1).to(device)
            batch_t1 = torch.tensor(
                np.stack(patches_t1[i:i+batch_size]), dtype=torch.float32
            ).unsqueeze(1).to(device)

            pred, _, _ = model(batch_t0, batch_t1, t=t_value)
            pred_np = pred.squeeze(1).cpu().numpy()
            pred_patches.extend([pred_np[j] for j in range(pred_np.shape[0])])

    # Stitch patches back into full frame
    stitched  = stitch_patches(pred_patches, positions, H, W, PATCH_SIZE)
    bt_output = denormalise(stitched)

    # Restore NaN mask from input (areas with no satellite coverage)
    nan_mask = np.isnan(arr_t0) & np.isnan(arr_t1)
    bt_output = np.where(nan_mask, np.nan, bt_output)

    return bt_output.astype(np.float32)


def save_h5(bt_array, t_value, t0_path, t1_path, output_dir):
    """
    Save interpolated frame as HDF5 matching INSAT-3DS structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"INSAT3DS_TIR_interp_t{t_value:.3f}_{timestamp}.h5"
    out_path  = os.path.join(output_dir, filename)

    with h5py.File(out_path, "w") as f:
        # Main data
        ds = f.create_dataset(
            "IMG_TIR1", data=bt_array,
            compression="gzip", compression_opts=4
        )
        ds.attrs["units"]       = "Kelvin"
        ds.attrs["long_name"]   = "Brightness Temperature 10.8 um"
        ds.attrs["valid_min"]   = 180.0
        ds.attrs["valid_max"]   = 320.0

        # Metadata
        f.attrs["satellite"]          = "INSAT-3DS"
        f.attrs["instrument"]         = "IMAGER"
        f.attrs["channel"]            = "TIR1"
        f.attrs["wavelength_um"]      = 10.8
        f.attrs["interpolated"]       = True
        f.attrs["t_parameter"]        = t_value
        f.attrs["interpolation_model"]= "RIFE-TIR-v1"
        f.attrs["source_frame_t0"]    = os.path.basename(t0_path)
        f.attrs["source_frame_t1"]    = os.path.basename(t1_path)
        f.attrs["created"]            = timestamp

    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def run_inference(t0_path, t1_path, mode="3x"):
    print("=" * 60)
    print("RIFE-TIR Inference Pipeline")
    print("=" * 60)
    print(f"T0     : {os.path.basename(t0_path)}")
    print(f"T1     : {os.path.basename(t1_path)}")
    print(f"Mode   : {mode} upscaling")
    print(f"Device : {DEVICE}")
    print()

    # Load model
    device = torch.device(DEVICE)
    model  = RIFE_TIR().to(device)
    ckpt   = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Model loaded from epoch {ckpt['epoch']}\n")

    # Load input frames
    arr_t0 = np.load(t0_path).astype(np.float32)
    arr_t1 = np.load(t1_path).astype(np.float32)
    print(f"Frame shape : {arr_t0.shape}")
    print(f"T0 BT range : {np.nanmin(arr_t0):.1f} K – {np.nanmax(arr_t0):.1f} K")
    print(f"T1 BT range : {np.nanmin(arr_t1):.1f} K – {np.nanmax(arr_t1):.1f} K\n")

    # Run interpolation for each t value
    t_values  = MODES[mode]
    outputs   = []

    for t_val in t_values:
        print(f"Interpolating t={t_val:.3f}...")
        bt_pred  = interpolate_pair(model, arr_t0, arr_t1, t_val, device)
        out_path = save_h5(bt_pred, t_val, t0_path, t1_path, OUTPUT_DIR)
        outputs.append((t_val, bt_pred, out_path))
        print(f"  BT range : {np.nanmin(bt_pred):.1f} K – {np.nanmax(bt_pred):.1f} K")
        print(f"  Saved    : {os.path.basename(out_path)}\n")

    # Summary
    print("=" * 60)
    print(f"Generated {len(outputs)} interpolated frame(s)")
    print(f"Mode: 30-min gap → {30 // int(mode[0])}-min effective cadence")
    print("=" * 60)

    return outputs


def visualise_results(arr_t0, arr_t1, outputs, save_path):
    """Save a side-by-side comparison image."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_interp = len(outputs)
    n_cols   = 2 + n_interp
    fig = plt.figure(figsize=(5 * n_cols, 5))
    gs  = gridspec.GridSpec(1, n_cols, figure=fig)

    vmin, vmax = 200, 310
    cmap = "gray_r"

    def plot_frame(ax, arr, title):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        return im

    ax0 = fig.add_subplot(gs[0])
    plot_frame(ax0, arr_t0, "T0 (real)")

    for i, (t_val, bt_pred, _) in enumerate(outputs):
        ax = fig.add_subplot(gs[i + 1])
        plot_frame(ax, bt_pred, f"t={t_val:.2f} (interpolated)")

    ax1 = fig.add_subplot(gs[-1])
    im  = plot_frame(ax1, arr_t1, "T1 (real)")

    # Shared colorbar
    fig.colorbar(im, ax=fig.get_axes(), shrink=0.8, label="Brightness Temperature (K)")
    plt.suptitle("RIFE-TIR: Temporal Interpolation of Thermal IR Frames",
                 fontsize=13, fontweight="bold")
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Visualisation saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0",   required=False, help="Path to T0 .npy grid file")
    parser.add_argument("--t1",   required=False, help="Path to T1 .npy grid file")
    parser.add_argument("--mode", default="3x",   help="2x, 3x, or 4x upscaling")
    args = parser.parse_args()

    # Default to first two val files if no args given
    if not args.t0 or not args.t1:
        import glob, json
        with open(r"D:\satellite-interp\data\triplet_index.json") as f:
            triplets = json.load(f)
        val_triplets = triplets[int(len(triplets)*0.8):]
        args.t0 = val_triplets[0]["t0"]
        args.t1 = val_triplets[0]["t1"]
        print(f"No args given — using first val triplet pair\n")

    outputs = run_inference(args.t0, args.t1, args.mode)

    # Visualise
    arr_t0 = np.load(args.t0)
    arr_t1 = np.load(args.t1)
    viz_path = os.path.join(OUTPUT_DIR, "interpolation_comparison.png")
    visualise_results(arr_t0, arr_t1, outputs, viz_path)