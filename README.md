# RIFE-TIR — Satellite Imagery Temporal Interpolation

ISRO Hackathon submission — optical flow deep learning model for INSAT-3DS
thermal IR frame interpolation.

## Results
| Metric | Baseline (Linear) | RIFE-TIR | Target |
|--------|------------------|----------|--------|
| SSIM   | 0.9137           | 0.9733   | ≥ 0.90 |
| PSNR   | 38.60 dB         | 43.07 dB | ≥ 35.0 |
| MSE    | 7.95 K²          | 1.95 K²  | ≤ 2.0  |
| FSIM   | 0.9720           | 0.9912   | ≥ 0.92 |

## What it does
Reduces INSAT-3DS temporal gap from **30 minutes → 10 minutes** (3x improvement)
without any hardware modification.

## Pipeline
1. `data/download_goes19.py` — fetch GOES-19 training data from NOAA S3
2. `data/calibrate.py` — radiance → brightness temperature (Kelvin)
3. `data/coregister.py` — reproject to common lat/lon grid
4. `data/dataset.py` — patch extraction + PyTorch dataset
5. `model/ifnet.py` — implicit flow estimator
6. `model/gridnet.py` — refinement network
7. `model/rife_tir.py` — complete model + physics-aware loss
8. `train.py` — training loop with early stopping
9. `validate.py` — SSIM / PSNR / MSE / FSIM metrics
10. `inference.py` — generate interpolated .h5 frames

## Key differentiator
Physics-aware loss function with four atmospheric constraints:
- Thermal continuity (max 2 K/min cooling rate)
- Mass conservation (cloud area bounded)
- Spatial smoothness (no checkerboard artefacts)
- Flow divergence (near-incompressible horizontal flow)

## Run inference
python inference.py --t0 frame_t0.npy --t1 frame_t1.npy --mode 3x