"""
validate.py
Computes SSIM, PSNR, MSE, FSIM for interpolated frames vs ground truth.
Generates a full validation report.
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.rife_tir import RIFE_TIR
from data.dataset import SatelliteTripletDataset, load_index, denormalise

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "index_file":  r"D:\satellite-interp\data\triplet_index.json",
    "ckpt_path":   r"D:\satellite-interp\outputs\checkpoints\best.pth",
    "report_file": r"D:\satellite-interp\outputs\validation_report.json",
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size":  4,
}

BT_MIN = 180.0
BT_MAX = 320.0


# ── Metric functions ──────────────────────────────────────────────────────────

def compute_mse(pred, target):
    """Mean Squared Error in K² (Kelvin squared)."""
    pred_bt   = denormalise(pred.cpu().numpy())
    target_bt = denormalise(target.cpu().numpy())
    return float(np.mean((pred_bt - target_bt) ** 2))


def compute_psnr(pred, target):
    """Peak Signal-to-Noise Ratio in dB. MAX = 140K range."""
    mse = compute_mse(pred, target)
    if mse < 1e-10:
        return 100.0
    max_val = BT_MAX - BT_MIN   # 140 K
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim_metric(pred, target):
    """SSIM in [0,1] — higher is better."""
    import torch.nn.functional as F
    C1, C2 = 0.01**2, 0.03**2
    mu_p  = F.avg_pool2d(pred,   3, 1, 1)
    mu_t  = F.avg_pool2d(target, 3, 1, 1)
    mu_pp = F.avg_pool2d(pred*pred,     3, 1, 1)
    mu_tt = F.avg_pool2d(target*target, 3, 1, 1)
    mu_pt = F.avg_pool2d(pred*target,   3, 1, 1)
    sig_p  = mu_pp - mu_p*mu_p
    sig_t  = mu_tt - mu_t*mu_t
    sig_pt = mu_pt - mu_p*mu_t
    ssim = ((2*mu_p*mu_t+C1)*(2*sig_pt+C2)) / \
           ((mu_p**2+mu_t**2+C1)*(sig_p+sig_t+C2))
    return float(ssim.mean().item())


def compute_fsim(pred, target):
    """
    Feature Similarity Index — uses phase congruency proxy via
    gradient magnitude similarity (GMS). Correlates strongly with FSIM.
    """
    import torch.nn.functional as F

    def gradient_magnitude(x):
        sobel_x = torch.tensor(
            [[-1,0,1],[-2,0,2],[-1,0,1]],
            dtype=torch.float32, device=x.device
        ).view(1,1,3,3)
        sobel_y = torch.tensor(
            [[-1,-2,-1],[0,0,0],[1,2,1]],
            dtype=torch.float32, device=x.device
        ).view(1,1,3,3)
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    gm_pred   = gradient_magnitude(pred)
    gm_target = gradient_magnitude(target)

    T = 0.05   # small constant
    gms = (2*gm_pred*gm_target + T) / (gm_pred**2 + gm_target**2 + T)
    return float(gms.mean().item())


def compute_baseline_metrics(t0, t1, gt):
    """Compute metrics for naive linear blend — our comparison baseline."""
    blend = 0.5 * t0 + 0.5 * t1
    return {
        "mse":  compute_mse(blend, gt),
        "psnr": compute_psnr(blend, gt),
        "ssim": compute_ssim_metric(blend, gt),
        "fsim": compute_fsim(blend, gt),
    }


# ── Main validation ───────────────────────────────────────────────────────────

def validate():
    print("=" * 60)
    print("RIFE-TIR Validation Report")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device(CONFIG["device"])
    model  = RIFE_TIR().to(device)

    ckpt = torch.load(CONFIG["ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"Device: {CONFIG['device']}\n")

    # ── Validation dataset ────────────────────────────────────────────────────
    triplets = load_index()
    split    = int(len(triplets) * 0.8)
    val_t    = triplets[split:]
    val_ds   = SatelliteTripletDataset(val_t, augment=False)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=0)

    print(f"Validation patches: {len(val_ds)}\n")

    # ── Accumulate metrics ────────────────────────────────────────────────────
    model_metrics = {"mse": [], "psnr": [], "ssim": [], "fsim": []}
    base_metrics  = {"mse": [], "psnr": [], "ssim": [], "fsim": []}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            t0, t1, gt, t_param = [x.to(device) for x in batch]

            # Model prediction
            pred, _, _ = model(t0, t1, t=t_param.mean().item())

            # Per-sample metrics
            for i in range(pred.shape[0]):
                p  = pred[i:i+1]
                g  = gt[i:i+1]
                f0 = t0[i:i+1]
                f1 = t1[i:i+1]

                model_metrics["mse" ].append(compute_mse(p, g))
                model_metrics["psnr"].append(compute_psnr(p, g))
                model_metrics["ssim"].append(compute_ssim_metric(p, g))
                model_metrics["fsim"].append(compute_fsim(p, g))

                bm = compute_baseline_metrics(f0, f1, g)
                for k in base_metrics:
                    base_metrics[k].append(bm[k])

    # ── Compute averages ──────────────────────────────────────────────────────
    def avg(lst): return float(np.mean(lst))

    results = {
        "model": {k: round(avg(v), 4) for k, v in model_metrics.items()},
        "baseline_linear_blend": {k: round(avg(v), 4) for k, v in base_metrics.items()},
    }

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Metric':<12} {'Linear Blend':>14} {'RIFE-TIR':>12} {'Improvement':>14}")
    print("-" * 60)

    improvements = {}
    for metric in ["ssim", "psnr", "mse", "fsim"]:
        b = results["baseline_linear_blend"][metric]
        m = results["model"][metric]

        if metric == "mse":
            imp = f"{((b-m)/b*100):+.1f}%"
            better = m < b
        else:
            imp = f"{((m-b)/b*100):+.1f}%"
            better = m > b

        marker = "✓" if better else "✗"
        print(f"{metric.upper():<12} {b:>14.4f} {m:>12.4f} {imp:>13} {marker}")
        improvements[metric] = imp

    print("=" * 60)

    # ── Targets check ─────────────────────────────────────────────────────────
    targets = {"ssim": 0.90, "psnr": 35.0, "mse": 2.0, "fsim": 0.92}
    print("\nTarget check:")
    all_pass = True
    for metric, target in targets.items():
        val = results["model"][metric]
        if metric == "mse":
            passed = val <= target
            print(f"  {metric.upper():<6} {val:.4f} {'<=' if passed else '>'} {target} {'✓ PASS' if passed else '✗ FAIL'}")
        else:
            passed = val >= target
            print(f"  {metric.upper():<6} {val:.4f} {'>=' if passed else '<'} {target} {'✓ PASS' if passed else '✗ FAIL'}")
        if not passed:
            all_pass = False

    print(f"\nOverall: {'✓ ALL TARGETS MET' if all_pass else '✗ Some targets missed'}")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "checkpoint_epoch": ckpt["epoch"],
        "num_val_patches":  len(val_ds),
        "metrics":          results,
        "targets":          targets,
        "all_targets_met":  all_pass,
    }
    os.makedirs(os.path.dirname(CONFIG["report_file"]), exist_ok=True)
    with open(CONFIG["report_file"], "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {CONFIG['report_file']}")


if __name__ == "__main__":
    validate()