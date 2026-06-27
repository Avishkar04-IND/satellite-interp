"""
export_demo_files.py
Exports sample .npy files from validation set for the Gradio demo.
"""
import numpy as np, json, os

with open(r"D:\satellite-interp\data\triplet_index.json") as f:
    triplets = json.load(f)

val = triplets[int(len(triplets)*0.8):]
t0  = np.load(val[0]["t0"])
t1  = np.load(val[0]["t1"])

os.makedirs("outputs/demo_files", exist_ok=True)
np.save("outputs/demo_files/demo_T0.npy", t0)
np.save("outputs/demo_files/demo_T1.npy", t1)

print("Demo files saved:")
print(f"  T0: outputs/demo_files/demo_T0.npy  {t0.shape}")
print(f"  T1: outputs/demo_files/demo_T1.npy  {t1.shape}")
print(f"  T0 BT range: {float(np.nanmin(t0)):.1f} – {float(np.nanmax(t0)):.1f} K")