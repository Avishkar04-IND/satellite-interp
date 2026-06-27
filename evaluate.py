import matplotlib.pyplot as plt
import numpy as np

pred = np.load(r"outputs\pred_1311.npy")

plt.figure(figsize=(10, 6))

plt.imshow(pred, cmap="plasma_r", origin="upper")
plt.colorbar(label="Brightness Temperature (K)")
plt.title("Predicted GOES F1 (13:11)")
plt.axis("off")

# Keep the original aspect ratio
plt.gca().set_aspect('equal')

plt.tight_layout()
plt.show()