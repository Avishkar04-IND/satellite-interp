import numpy as np
import matplotlib.pyplot as plt

# Change this path to the file you want to view
file_path = r"data\goes19_raw\2026\013\13\OR_ABI-L1b-RadC-M6C13_G19_s20260131311178_e20260131313563_c20260131314054.nc"

img = np.load(file_path)

print("Shape:", img.shape)
print("Min:", img.min())
print("Max:", img.max())

plt.figure(figsize=(10, 6))
plt.imshow(img, cmap="plasma_r")
plt.colorbar(label="Brightness Temperature (K)")
plt.title("GOES-19 Brightness Temperature")
plt.show()