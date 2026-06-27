import numpy as np

real = np.load(r"data\goes19_bt\2026\013\13\OR_ABI-L1b-RadC-M6C13_G19_s20260131311178_e20260131313563_c20260131314054_BT.npy")
pred = np.load(r"outputs\pred_1311.npy")

print("REAL")
print("Shape :", real.shape)
print("NaNs  :", np.isnan(real).sum())
print("Min   :", np.nanmin(real))
print("Max   :", np.nanmax(real))

print()

print("PRED")
print("Shape :", pred.shape)
print("NaNs  :", np.isnan(pred).sum())
print("Min   :", np.nanmin(pred))
print("Max   :", np.nanmax(pred))