"""
calibrate.py
Converts raw GOES-19 radiance values → Brightness Temperature (BT) in Kelvin.
Uses the Planck constants embedded in each NetCDF file's metadata.
"""

import netCDF4 as nc
import numpy as np
import os
import glob
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR  = r"D:\satellite-interp\data\goes19_raw"
OUT_DIR  = r"D:\satellite-interp\data\goes19_bt"
BT_MIN   = 180.0   # Kelvin — deep cold cloud tops
BT_MAX   = 320.0   # Kelvin — hot land surface

# ── Core conversion ───────────────────────────────────────────────────────────
def radiance_to_bt(radiance, fk1, fk2, bc1, bc2):
    """
    Apply Planck function to convert radiance → Brightness Temperature.
    Formula from GOES-R Product User Guide (PUG), Vol. 3.

    BT = (fk2 / (log(fk1 / radiance + 1) - bc1)) / bc2
    """
    # Avoid log(0) or negative values
    radiance = np.where(radiance > 0, radiance, np.nan)
    bt = (fk2 / (np.log(fk1 / radiance + 1.0) - bc1)) / bc2
    return bt.astype(np.float32)

def process_file(nc_path, out_dir):
    """
    Read one GOES-19 L1b NetCDF file, convert to BT, save as .npy
    """
    filename = os.path.basename(nc_path)
    out_name = filename.replace(".nc", "_BT.npy")
    out_path = os.path.join(out_dir, out_name)

    if os.path.exists(out_path):
        return out_path, None  # already processed

    try:
        with nc.Dataset(nc_path, "r") as ds:
            # Read raw radiance (scaled integer → float via scale/offset)
            rad_var   = ds.variables["Rad"]
            radiance  = rad_var[:].astype(np.float32)

            # Read Planck calibration constants from file metadata
            fk1 = float(ds.variables["planck_fk1"][:])
            fk2 = float(ds.variables["planck_fk2"][:])
            bc1 = float(ds.variables["planck_bc1"][:])
            bc2 = float(ds.variables["planck_bc2"][:])

            # Convert
            bt = radiance_to_bt(radiance, fk1, fk2, bc1, bc2)

            # Basic quality check
            valid_mask = (bt >= BT_MIN) & (bt <= BT_MAX)
            bt = np.where(valid_mask, bt, np.nan)

            # Save
            os.makedirs(out_dir, exist_ok=True)
            np.save(out_path, bt)

            stats = {
                "min":   float(np.nanmin(bt)),
                "max":   float(np.nanmax(bt)),
                "mean":  float(np.nanmean(bt)),
                "valid": float(np.sum(valid_mask) / bt.size * 100)
            }
            return out_path, stats

    except Exception as e:
        print(f"  ERROR processing {filename}: {e}")
        return None, None

def calibrate_all():
    """Walk the raw directory and calibrate every .nc file found."""
    # Find all nc files recursively
    pattern   = os.path.join(RAW_DIR, "**", "*.nc")
    nc_files  = sorted(glob.glob(pattern, recursive=True))

    if not nc_files:
        print(f"No .nc files found in {RAW_DIR}")
        return

    print(f"Found {len(nc_files)} files to calibrate\n")

    success = 0
    for nc_path in tqdm(nc_files, desc="Calibrating"):
        # Mirror the subfolder structure in output dir
        rel_path = os.path.relpath(os.path.dirname(nc_path), RAW_DIR)
        out_dir  = os.path.join(OUT_DIR, rel_path)

        out_path, stats = process_file(nc_path, out_dir)

        if stats:
            success += 1

    print(f"\nCalibration complete.")
    print(f"Successfully processed : {success} / {len(nc_files)} files")
    print(f"Output saved to        : {OUT_DIR}")

def inspect_one():
    """Quick sanity check — print BT stats for the first file found."""
    pattern  = os.path.join(RAW_DIR, "**", "*.nc")
    nc_files = sorted(glob.glob(pattern, recursive=True))

    if not nc_files:
        print("No files found.")
        return

    first = nc_files[0]
    print(f"Inspecting: {os.path.basename(first)}")

    _, stats = process_file(first, os.path.join(OUT_DIR, "inspect"))

    if stats:
        print(f"\nBrightness Temperature stats:")
        print(f"  Min  : {stats['min']:.2f} K")
        print(f"  Max  : {stats['max']:.2f} K")
        print(f"  Mean : {stats['mean']:.2f} K")
        print(f"  Valid pixels: {stats['valid']:.1f}%")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("GOES-19 Radiance → Brightness Temperature Calibration")
    print("=" * 60)

    # First inspect one file to confirm it works
    print("\n--- Sanity check on first file ---")
    inspect_one()

    # Then process all files
    print("\n--- Processing all files ---")
    calibrate_all()