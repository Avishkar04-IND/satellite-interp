"""
coregister.py
Reprojects all calibrated GOES-19 BT arrays onto a common
equirectangular grid (0.04 degree per pixel ~ 4 km at equator).
This ensures GOES-19 and INSAT-3DS can be compared pixel-by-pixel.
"""

import numpy as np
import netCDF4 as nc
import os
import glob
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR   = r"D:\satellite-interp\data\goes19_raw"
BT_DIR    = r"D:\satellite-interp\data\goes19_bt"
OUT_DIR   = r"D:\satellite-interp\data\goes19_grid"

# Common output grid — 0.04 degree resolution
# Covers GOES-19 CONUS domain approximately
LAT_MIN, LAT_MAX = 15.0,  55.0    # degrees North
LON_MIN, LON_MAX = -135.0, -60.0  # degrees West

GRID_RES  = 0.04   # degrees per pixel

# ── Build target grid ─────────────────────────────────────────────────────────
TARGET_LATS = np.arange(LAT_MAX, LAT_MIN, -GRID_RES, dtype=np.float32)
TARGET_LONS = np.arange(LON_MIN, LON_MAX,  GRID_RES, dtype=np.float32)
GRID_H      = len(TARGET_LATS)
GRID_W      = len(TARGET_LONS)

def get_lat_lon(nc_path):
    """
    Extract latitude and longitude arrays from a GOES-19 NetCDF file.
    GOES-19 uses a fixed-grid geostationary projection — we convert
    it to geographic lat/lon using the projection parameters in the file.
    """
    with nc.Dataset(nc_path, "r") as ds:
        # Read projection info
        proj_var = ds.variables["goes_imager_projection"]
        lon_origin    = proj_var.longitude_of_projection_origin
        r_eq          = proj_var.semi_major_axis
        r_pol         = proj_var.semi_minor_axis
        sat_height    = proj_var.perspective_point_height + r_eq

        # Fixed grid coordinates (radians)
        x = ds.variables["x"][:].astype(np.float64)  # scan angle EW
        y = ds.variables["y"][:].astype(np.float64)  # scan angle NS

        # Meshgrid
        x2d, y2d = np.meshgrid(x, y)

        # GOES projection formula (from PUG documentation)
        a = np.sin(x2d)**2 + (np.cos(x2d)**2 * (np.cos(y2d)**2 + (r_eq/r_pol)**2 * np.sin(y2d)**2))
        b = -2 * sat_height * np.cos(x2d) * np.cos(y2d)
        c = sat_height**2 - r_eq**2

        discriminant = b**2 - 4*a*c
        valid = discriminant >= 0

        rs = np.full_like(a, np.nan)
        rs[valid] = (-b[valid] - np.sqrt(discriminant[valid])) / (2 * a[valid])

        sx = rs * np.cos(x2d) * np.cos(y2d)
        sy = -rs * np.sin(x2d)
        sz = rs * np.cos(x2d) * np.sin(y2d)

        lat = np.degrees(np.arctan((r_eq/r_pol)**2 * sz / np.sqrt((sat_height - sx)**2 + sy**2)))
        lon = np.degrees(np.arctan(sy / (sat_height - sx))) + lon_origin

        return lat.astype(np.float32), lon.astype(np.float32)

def reproject(bt, src_lat, src_lon):
    """
    Reproject a BT array from native GOES grid → common lat/lon grid.
    Uses griddata for scattered interpolation — handles non-regular grids.
    """
    from scipy.interpolate import griddata

    # Flatten source arrays
    src_lat_flat = src_lat.ravel()
    src_lon_flat = src_lon.ravel()
    bt_flat      = bt.ravel()

    # Remove NaN points
    valid = (~np.isnan(bt_flat)) & (~np.isnan(src_lat_flat)) & (~np.isnan(src_lon_flat))

    if valid.sum() < 1000:
        print("  Warning: too few valid pixels, skipping")
        return None

    src_points = np.column_stack([src_lat_flat[valid], src_lon_flat[valid]])
    src_values = bt_flat[valid]

    # Build target meshgrid
    lon_grid, lat_grid = np.meshgrid(TARGET_LONS, TARGET_LATS)
    target_points      = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    # Filter target points that fall within source coverage
    lat_min_src = src_lat_flat[valid].min()
    lat_max_src = src_lat_flat[valid].max()
    lon_min_src = src_lon_flat[valid].min()
    lon_max_src = src_lon_flat[valid].max()

    in_bounds = (
        (target_points[:, 0] >= lat_min_src) &
        (target_points[:, 0] <= lat_max_src) &
        (target_points[:, 1] >= lon_min_src) &
        (target_points[:, 1] <= lon_max_src)
    )

    result = np.full(GRID_H * GRID_W, np.nan, dtype=np.float32)

    if in_bounds.sum() > 0:
        interpolated = griddata(
            src_points,
            src_values,
            target_points[in_bounds],
            method="linear"
        )
        result[in_bounds] = interpolated

    return result.reshape(GRID_H, GRID_W)

def process_pair(nc_path, bt_path, out_dir):
    """
    Load lat/lon from .nc file, load BT from .npy, reproject, save.
    """
    filename = os.path.basename(bt_path)
    out_name = filename.replace("_BT.npy", "_GRID.npy")
    out_path = os.path.join(out_dir, out_name)

    if os.path.exists(out_path):
        return out_path

    try:
        bt      = np.load(bt_path)
        lat, lon = get_lat_lon(nc_path)
        gridded  = reproject(bt, lat, lon)

        if gridded is None:
            return None

        os.makedirs(out_dir, exist_ok=True)
        np.save(out_path, gridded)
        return out_path

    except Exception as e:
        print(f"  ERROR: {os.path.basename(nc_path)} — {e}")
        return None

def coregister_all():
    """Match every BT file to its source .nc and reproject."""
    bt_files = sorted(glob.glob(
        os.path.join(BT_DIR, "**", "*_BT.npy"), recursive=True
    ))

    if not bt_files:
        print(f"No BT files found in {BT_DIR}")
        return

    print(f"Found {len(bt_files)} BT files to reproject\n")
    success = 0

    for bt_path in tqdm(bt_files, desc="Reprojecting"):
        # Reconstruct original .nc filename from _BT.npy name
        bt_name  = os.path.basename(bt_path)
        nc_name  = bt_name.replace("_BT.npy", ".nc")

        # Mirror folder structure back to raw dir
        rel      = os.path.relpath(os.path.dirname(bt_path), BT_DIR)
        raw_dir  = os.path.join(RAW_DIR, rel)
        out_dir  = os.path.join(OUT_DIR, rel)
        nc_path  = os.path.join(raw_dir, nc_name)

        # Debug — print first pair to confirm match
        if success == 0 and bt_files.index(bt_path) == 0:
            print(f"\n  BT file : {bt_name}")
            print(f"  NC file : {nc_name}")
            print(f"  NC exists: {os.path.exists(nc_path)}\n")

        if not os.path.exists(nc_path):
            print(f"  Missing NC: {nc_path}")
            continue

        result = process_pair(nc_path, bt_path, out_dir)
        if result:
            success += 1

    print(f"\nReprojection complete.")
    print(f"Processed : {success} / {len(bt_files)} files")
    print(f"Grid size : {GRID_H} x {GRID_W} pixels")
    print(f"Resolution: {GRID_RES} degrees/pixel (~4 km)")
    print(f"Output    : {OUT_DIR}")

def inspect_grid():
    """Print stats for first reprojected file."""
    files = sorted(glob.glob(
        os.path.join(OUT_DIR, "**", "*_GRID.npy"), recursive=True
    ))
    if not files:
        print("No grid files yet.")
        return

    arr = np.load(files[0])
    print(f"\nGrid file : {os.path.basename(files[0])}")
    print(f"Shape     : {arr.shape}")
    print(f"Min BT    : {np.nanmin(arr):.2f} K")
    print(f"Max BT    : {np.nanmax(arr):.2f} K")
    print(f"Mean BT   : {np.nanmean(arr):.2f} K")
    print(f"NaN pixels: {np.isnan(arr).sum()} / {arr.size}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("GOES-19 Co-registration → Common Lat/Lon Grid")
    print(f"Target grid : {GRID_H} x {GRID_W} pixels")
    print(f"Coverage    : {LAT_MIN}°N–{LAT_MAX}°N, {LON_MIN}°E–{LON_MAX}°E")
    print("=" * 60)

    coregister_all()

    print("\n--- Grid inspection ---")
    inspect_grid()