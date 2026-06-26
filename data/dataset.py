"""
dataset.py
Slices reprojected GOES-19 grid files into (T0, T1, T_mid) patch triplets
for training the frame interpolation model.
Each triplet = two boundary frames + one ground truth intermediate frame.
"""

import numpy as np
import os
import glob
import json
from torch.utils.data import Dataset
import torch

# ── Configuration ─────────────────────────────────────────────────────────────
GRID_DIR    = r"D:\satellite-interp\data\goes19_grid"
INDEX_FILE  = r"D:\satellite-interp\data\triplet_index.json"

PATCH_SIZE  = 128          # pixels — 128x128 fits T4 GPU comfortably
STRIDE      = 64           # 50% overlap between patches
BT_MIN      = 180.0        # Kelvin — normalisation range
BT_MAX      = 320.0        # Kelvin — normalisation range
MIN_VALID   = 0.70         # minimum fraction of non-NaN pixels in a patch

# ── Normalisation helpers ──────────────────────────────────────────────────────
def normalise(bt_array):
    """Scale BT from [180K, 320K] → [0, 1]"""
    return np.clip((bt_array - BT_MIN) / (BT_MAX - BT_MIN), 0.0, 1.0)

def denormalise(norm_array):
    """Scale [0, 1] back → BT in Kelvin"""
    return norm_array * (BT_MAX - BT_MIN) + BT_MIN

# ── Triplet index builder ──────────────────────────────────────────────────────
def extract_timestamp(filename):
    """
    Extract start timestamp from GOES-19 filename.
    Example: ...G19_s20251520601171... → '20251520601171'
    """
    base = os.path.basename(filename)
    parts = base.split("_")
    for p in parts:
        if p.startswith("s2025"):
            return p[1:]   # remove leading 's'
    return None

def build_triplet_index():
    """
    Find all grid files, sort by timestamp, group into consecutive triplets.
    A valid triplet = three files where spacing is approximately equal
    (T0, T_mid, T1) where T_mid is the ground truth intermediate frame.
    """
    grid_files = sorted(glob.glob(
        os.path.join(GRID_DIR, "**", "*_GRID.npy"), recursive=True
    ))

    if len(grid_files) < 3:
        print(f"Need at least 3 grid files, found {len(grid_files)}")
        return []

    print(f"Found {len(grid_files)} grid files")

    # Sort by timestamp
    timestamped = []
    for f in grid_files:
        ts = extract_timestamp(f)
        if ts:
            timestamped.append((ts, f))

    timestamped.sort(key=lambda x: x[0])

    # Build triplets: (T0, T_mid, T1) from consecutive files
    triplets = []
    for i in range(len(timestamped) - 2):
        ts0,  f0   = timestamped[i]
        ts_mid, f_mid = timestamped[i + 1]
        ts1,  f1   = timestamped[i + 2]

        triplets.append({
            "t0":    f0,
            "t_mid": f_mid,
            "t1":    f1,
            "ts0":   ts0,
            "ts_mid": ts_mid,
            "ts1":   ts1,
            "t_param": 0.5      # midpoint between T0 and T1
        })

    print(f"Built {len(triplets)} triplets")
    return triplets

def save_index(triplets):
    """Save triplet index to JSON for reuse."""
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    with open(INDEX_FILE, "w") as f:
        json.dump(triplets, f, indent=2)
    print(f"Index saved to {INDEX_FILE}")

def load_index():
    """Load triplet index from JSON."""
    with open(INDEX_FILE, "r") as f:
        return json.load(f)

# ── Patch extraction ───────────────────────────────────────────────────────────
def extract_patches(array, patch_size, stride):
    """
    Slice a 2D array into overlapping patches.
    Returns list of (row_start, col_start) positions.
    """
    H, W = array.shape
    positions = []
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = array[r:r+patch_size, c:c+patch_size]
            # Skip patches with too many NaN values
            valid_frac = (~np.isnan(patch)).mean()
            if valid_frac >= MIN_VALID:
                positions.append((r, c))
    return positions

# ── PyTorch Dataset ────────────────────────────────────────────────────────────
class SatelliteTripletDataset(Dataset):
    """
    PyTorch Dataset that returns (frame_t0, frame_t1, frame_tmid, t_param) tensors.
    Each item is a 128x128 patch from a triplet of satellite frames.
    """

    def __init__(self, triplets, patch_size=PATCH_SIZE, stride=STRIDE, augment=False):
        self.triplets   = triplets
        self.patch_size = patch_size
        self.stride     = stride
        self.augment    = augment
        self.items      = []   # list of (triplet_idx, row, col)
        self._build_items()

    def _build_items(self):
        """Pre-compute all valid patch positions across all triplets."""
        print("Building patch index...")
        for idx, triplet in enumerate(self.triplets):
            try:
                # Load just T0 to find valid patch positions
                t0 = np.load(triplet["t0"])
                positions = extract_patches(t0, self.patch_size, self.stride)
                for (r, c) in positions:
                    self.items.append((idx, r, c))
            except Exception as e:
                print(f"  Skipping triplet {idx}: {e}")

        print(f"Total patches available: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def _load_patch(self, filepath, r, c):
        """Load one patch from a grid file, normalise, fill NaN."""
        arr   = np.load(filepath)
        patch = arr[r:r+self.patch_size, c:c+self.patch_size].copy()
        patch = normalise(patch)
        # Fill remaining NaN with 0.5 (mid-range neutral value)
        patch = np.where(np.isnan(patch), 0.5, patch)
        return patch.astype(np.float32)

    def _augment(self, t0, t1, tmid):
        """Apply consistent augmentation to all three frames."""
        # Horizontal flip
        if np.random.rand() > 0.5:
            t0   = np.fliplr(t0).copy()
            t1   = np.fliplr(t1).copy()
            tmid = np.fliplr(tmid).copy()
        # 90-degree rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4)
        if k > 0:
            t0   = np.rot90(t0,   k).copy()
            t1   = np.rot90(t1,   k).copy()
            tmid = np.rot90(tmid, k).copy()
        return t0, t1, tmid

    def __getitem__(self, index):
        triplet_idx, r, c = self.items[index]
        triplet = self.triplets[triplet_idx]

        t0   = self._load_patch(triplet["t0"],    r, c)
        t1   = self._load_patch(triplet["t1"],    r, c)
        tmid = self._load_patch(triplet["t_mid"], r, c)

        if self.augment:
            t0, t1, tmid = self._augment(t0, t1, tmid)

        t_param = float(triplet["t_param"])

        # Add channel dimension: (1, H, W)
        return (
            torch.from_numpy(t0  ).unsqueeze(0),
            torch.from_numpy(t1  ).unsqueeze(0),
            torch.from_numpy(tmid).unsqueeze(0),
            torch.tensor(t_param, dtype=torch.float32)
        )

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Building Satellite Triplet Dataset")
    print("=" * 60)

    # Build and save triplet index
    triplets = build_triplet_index()
    if not triplets:
        exit(1)

    save_index(triplets)

    # Train / val split by index (not random — preserves temporal order)
    split     = int(len(triplets) * 0.8)
    train_t   = triplets[:split]
    val_t     = triplets[split:]

    print(f"\nTrain triplets : {len(train_t)}")
    print(f"Val triplets   : {len(val_t)}")

    # Build datasets
    train_ds = SatelliteTripletDataset(train_t, augment=True)
    val_ds   = SatelliteTripletDataset(val_t,   augment=False)

    print(f"\nTrain patches  : {len(train_ds)}")
    print(f"Val patches    : {len(val_ds)}")

    # Test one batch
    print("\nTesting one sample...")
    t0, t1, tmid, t_param = train_ds[0]
    print(f"  t0    shape : {t0.shape}   min={t0.min():.3f} max={t0.max():.3f}")
    print(f"  t1    shape : {t1.shape}   min={t1.min():.3f} max={t1.max():.3f}")
    print(f"  tmid  shape : {tmid.shape} min={tmid.min():.3f} max={tmid.max():.3f}")
    print(f"  t_param     : {t_param:.3f}")
    print("\nDataset ready for training.")