"""
download_goes19.py
Downloads GOES-19 ABI Channel 13 (thermal IR) data from NOAA's public S3 bucket.
No AWS account needed — bucket is fully public.
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os
from datetime import datetime, timedelta
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
BUCKET_NAME = "noaa-goes19"
PRODUCT     = "ABI-L1b-RadC"       # RadC = CONUS, 5-min cadence
CHANNEL     = "C13"                 # Channel 13 = 10.3 µm thermal IR
SAVE_DIR    = r"D:\satellite-interp\data\goes19_raw"

# ── S3 Client (anonymous — no credentials needed) ────────────────────────────
s3 = boto3.client(
    "s3",
    config=Config(signature_version=UNSIGNED)
)

def list_files(year, day_of_year, hour):
    """List all C13 files for a given year/day/hour."""
    prefix = f"{PRODUCT}/{year}/{day_of_year:03d}/{hour:02d}/"
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

    if "Contents" not in response:
        print(f"  No files found for prefix: {prefix}")
        return []

    # Filter only Channel 13 files
    files = [
        obj["Key"] for obj in response["Contents"]
        if CHANNEL in obj["Key"]
    ]
    return files

def download_file(s3_key, save_dir):
    """Download a single file from S3 to local disk."""
    filename  = os.path.basename(s3_key)
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print(f"  Already exists, skipping: {filename}")
        return save_path

    os.makedirs(save_dir, exist_ok=True)
    print(f"  Downloading: {filename}")

    s3.download_file(BUCKET_NAME, s3_key, save_path)
    return save_path

def download_range(start_date, end_date, hours=None):
    """
    Download all C13 files between start_date and end_date.
    hours = list of hours to download e.g. [6, 7, 8] for 06:00-08:00 UTC
    If hours is None, downloads all 24 hours.
    """
    if hours is None:
        hours = list(range(24))

    current = start_date
    all_files = []

    while current <= end_date:
        year       = current.year
        day_of_year = current.timetuple().tm_yday

        print(f"\nDate: {current.strftime('%Y-%m-%d')} (Day {day_of_year})")

        for hour in hours:
            print(f"  Hour: {hour:02d}:00 UTC")
            save_dir = os.path.join(
                SAVE_DIR,
                f"{year}",
                f"{day_of_year:03d}",
                f"{hour:02d}"
            )

            files = list_files(year, day_of_year, hour)
            print(f"  Found {len(files)} C13 files")

            for f in tqdm(files, desc=f"    Day {day_of_year} Hr {hour:02d}"):
                path = download_file(f, save_dir)
                all_files.append(path)

        current += timedelta(days=1)

    return all_files

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("GOES-19 ABI Channel 13 Downloader")
    print("=" * 60)

    # Download 1 day of data, hours 6-8 UTC only (small test first)
    start = datetime(2025, 6, 1)
    end   = datetime(2025, 6, 1)   # same day = just 1 day
    hours = [6, 7, 8]              # 3 hours = ~36 files, ~500MB

    downloaded = download_range(start, end, hours=hours)

    print(f"\nDone. Downloaded {len(downloaded)} files.")
    print(f"Saved to: {SAVE_DIR}")