"""
app.py
Gradio demo for RIFE-TIR satellite frame interpolation.
Upload two .npy grid files → get interpolated 15-min frame.
"""

import gradio as gr
import numpy as np
import torch
import os, sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.rife_tir import RIFE_TIR
from data.dataset import normalise, denormalise

CKPT_PATH = r"D:\satellite-interp\outputs\checkpoints\best.pth"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading RIFE-TIR model...")
model = RIFE_TIR()
ckpt  = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Model ready (epoch {ckpt['epoch']})")


# ── Core interpolation ────────────────────────────────────────────────────────
def interpolate_frame(arr_t0, arr_t1, t_value):
    PATCH, STRIDE = 128, 64
    norm_t0 = np.where(np.isnan(arr_t0), 0.5, normalise(arr_t0))
    norm_t1 = np.where(np.isnan(arr_t1), 0.5, normalise(arr_t1))
    H, W    = norm_t0.shape
    output  = np.zeros((H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    window  = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)

    p0, p1, pos = [], [], []
    for r in range(0, H - PATCH + 1, STRIDE):
        for c in range(0, W - PATCH + 1, STRIDE):
            p0.append(norm_t0[r:r+PATCH, c:c+PATCH])
            p1.append(norm_t1[r:r+PATCH, c:c+PATCH])
            pos.append((r, c))

    with torch.no_grad():
        for i in range(0, len(p0), 16):
            b0 = torch.tensor(np.stack(p0[i:i+16])).unsqueeze(1)
            b1 = torch.tensor(np.stack(p1[i:i+16])).unsqueeze(1)
            pred, _, _ = model(b0, b1, t=t_value)
            pred_np = pred.squeeze(1).numpy()
            for j, (r, c) in enumerate(pos[i:i+16]):
                output [r:r+PATCH, c:c+PATCH] += pred_np[j] * window
                weights[r:r+PATCH, c:c+PATCH] += window

    weights = np.where(weights < 1e-6, 1.0, weights)
    return denormalise(output / weights).astype(np.float32)


def bt_to_image(arr, title):
    """Convert BT array to a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="gray_r", vmin=200, vmax=312, origin="upper")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Brightness Temp (K)")
    plt.tight_layout()
    return fig


# ── Gradio prediction function ────────────────────────────────────────────────
def predict(t0_file, t1_file, mode):
    if t0_file is None or t1_file is None:
        return None, None, None, "Please upload both T0 and T1 .npy files."

    arr_t0 = np.load(t0_file.name).astype(np.float32)
    arr_t1 = np.load(t1_file.name).astype(np.float32)

    t_map = {
        "2x — 15 min (t=0.5)":         [0.5],
        "3x — 10 min (t=0.333, 0.667)": [0.333, 0.667],
    }
    t_values = t_map[mode]

    # Always generate t=0.5 for display
    pred_15 = interpolate_frame(arr_t0, arr_t1, 0.5)

    # Stats
    stats = (
        f"T0 BT: {np.nanmin(arr_t0):.1f} – {np.nanmax(arr_t0):.1f} K\n"
        f"T1 BT: {np.nanmin(arr_t1):.1f} – {np.nanmax(arr_t1):.1f} K\n"
        f"Pred BT: {np.nanmin(pred_15):.1f} – {np.nanmax(pred_15):.1f} K\n"
        f"Mode: {mode}\n"
        f"Model: RIFE-TIR epoch {ckpt['epoch']}"
    )

    fig_t0   = bt_to_image(arr_t0,  "T0 — Input frame")
    fig_pred = bt_to_image(pred_15, f"Predicted midpoint (t=0.5)")
    fig_t1   = bt_to_image(arr_t1,  "T1 — Input frame")

    return fig_t0, fig_pred, fig_t1, stats


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="RIFE-TIR Satellite Interpolation", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🛰️ RIFE-TIR — Satellite Thermal IR Frame Interpolation
    ### ISRO Hackathon | Temporal Resolution Enhancement

    **Upload two consecutive satellite frames (as `.npy` files) →
    Model generates the intermediate frame**

    - Input: Two frames 30 minutes apart
    - Output: Synthesised intermediate frame
    - Physics-aware loss ensures atmospheric consistency
    """)

    with gr.Row():
        with gr.Column():
            t0_input = gr.File(label="T0 frame (.npy) — earlier time")
            t1_input = gr.File(label="T1 frame (.npy) — 30 min later")
            mode     = gr.Radio(
                choices=["2x — 15 min (t=0.5)",
                         "3x — 10 min (t=0.333, 0.667)"],
                value="2x — 15 min (t=0.5)",
                label="Upscaling mode"
            )
            run_btn  = gr.Button("Generate Intermediate Frame",
                                 variant="primary", size="lg")

        with gr.Column():
            stats_out = gr.Textbox(label="Frame statistics", lines=6)

    gr.Markdown("### Output")
    with gr.Row():
        out_t0   = gr.Plot(label="T0 — Input")
        out_pred = gr.Plot(label="Predicted midpoint")
        out_t1   = gr.Plot(label="T1 — Input")

    gr.Markdown("""
    ### Validation Results (controlled test set)
    | Metric | Linear Blend | RIFE-TIR | Improvement |
    |--------|-------------|----------|-------------|
    | SSIM   | 0.8481      | 0.9023   | +6.4% ✓    |
    | PSNR   | 31.74 dB    | 33.45 dB | +5.4% ✓    |
    | MSE    | 36.68 K²    | 28.99 K² | +21.0% ✓   |
    | FSIM   | 0.9403      | 0.9502   | +1.1% ✓    |

    ### Physics-aware constraints
    - Thermal continuity (max 2K/min cooling rate)
    - Mass conservation (cloud area bounded)
    - Flow divergence (near-incompressible atmosphere)
    - Spatial smoothness (no checkerboard artefacts)
    """)

    run_btn.click(
        predict,
        inputs=[t0_input, t1_input, mode],
        outputs=[out_t0, out_pred, out_t1, stats_out]
    )

    gr.Markdown("""
    ---
    **How to get demo files:** Run `python export_demo_files.py` to export
    sample `.npy` frames from the validation set.
    """)

if __name__ == "__main__":
    demo.launch(share=False)