"""
rife_tir.py
Complete RIFE-TIR model — connects IFNet + GridNet + physics-aware loss.
This is the main model class used for training and inference.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ifnet import IFNet
from model.gridnet import GridNet


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS-AWARE LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def charbonnier_loss(pred, target, eps=1e-6):
    """Robust L1-like loss — less sensitive to outliers than MSE."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))


def ssim_loss(pred, target):
    """1 - SSIM so it acts as a minimisable loss."""
    C1, C2 = 0.01**2, 0.03**2
    mu_p  = F.avg_pool2d(pred,   3, 1, 1)
    mu_t  = F.avg_pool2d(target, 3, 1, 1)
    mu_pp = F.avg_pool2d(pred   * pred,   3, 1, 1)
    mu_tt = F.avg_pool2d(target * target, 3, 1, 1)
    mu_pt = F.avg_pool2d(pred   * target, 3, 1, 1)
    sig_p  = mu_pp - mu_p * mu_p
    sig_t  = mu_tt - mu_t * mu_t
    sig_pt = mu_pt - mu_p * mu_t
    ssim_map = ((2*mu_p*mu_t + C1) * (2*sig_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2))
    return 1.0 - ssim_map.mean()


def thermal_continuity_loss(pred, t0, t1, t, max_rate=2.0, dt_minutes=5.0):
    """
    Penalise BT changes that exceed physically possible cooling/heating rates.
    max_rate : maximum observed atmospheric BT change rate in K/min
    dt_minutes: time gap between frames in minutes
    """
    bt_mean    = (1.0 - t) * t0 + t * t1
    deviation  = torch.abs(pred - bt_mean)
    max_allowed = max_rate * dt_minutes * min(t, 1.0 - t)
    violation  = torch.clamp(deviation - max_allowed, min=0.0)
    return violation.mean()


def mass_conservation_loss(pred, t0, t1, cold_thresh=0.43, tolerance=0.15):
    """
    Penalise large changes in cold cloud area between frames.
    cold_thresh=0.43 corresponds to ~240K in normalised [0,1] space.
    (240 - 180) / (320 - 180) = 0.43
    """
    mask_t0   = (t0   < cold_thresh).float()
    mask_t1   = (t1   < cold_thresh).float()
    mask_pred = (pred < cold_thresh).float()

    area_t0   = mask_t0.sum(dim=[-1, -2])
    area_t1   = mask_t1.sum(dim=[-1, -2])
    area_exp  = 0.5 * (area_t0 + area_t1)
    area_pred = mask_pred.sum(dim=[-1, -2])

    deviation = torch.abs(area_pred - area_exp) / (area_exp + 1e-6)
    violation = torch.clamp(deviation - tolerance, min=0.0)
    return violation.mean()


def spatial_smoothness_loss(pred):
    """Total variation — penalises checkerboard and high-frequency artefacts."""
    diff_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    diff_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return diff_x.abs().mean() + diff_y.abs().mean()


def flow_divergence_loss(flow):
    """
    Penalise large divergence in the flow field.
    Horizontal atmospheric flow is near-incompressible: du/dx + dv/dy ≈ 0
    """
    u = flow[:, 0:1, :, :]
    v = flow[:, 1:2, :, :]
    du_dx = F.pad(u[:, :, :, 1:] - u[:, :, :, :-1], (0, 1))
    dv_dy = F.pad(v[:, :, 1:, :] - v[:, :, :-1, :], (0, 0, 0, 1))
    divergence = du_dx + dv_dy
    return divergence.pow(2).mean()


def physics_aware_loss(pred, flow, t0, t1, gt, t):
    """
    Combined loss: pixel fidelity + four physics constraints.
    Returns total loss and a dict of individual components for logging.
    """
    # ── Pixel losses ──────────────────────────────────────────────────────────
    L_pixel = charbonnier_loss(pred, gt)
    L_ssim  = ssim_loss(pred, gt)

    # ── Physics constraints ───────────────────────────────────────────────────
    L_therm  = thermal_continuity_loss(pred, t0, t1, t.mean().item())
    L_mass   = mass_conservation_loss(pred, t0, t1)
    L_smooth = spatial_smoothness_loss(pred)
    L_div    = flow_divergence_loss(flow)

    # ── Weighted total ────────────────────────────────────────────────────────
    total = (
        0.80 * L_pixel  +
        0.10 * L_ssim   +
        0.05 * L_therm  +
        0.05 * L_mass   +
        0.01 * L_smooth +
        0.02 * L_div
    )

    components = {
        "pixel":  L_pixel.item(),
        "ssim":   L_ssim.item(),
        "therm":  L_therm.item(),
        "mass":   L_mass.item(),
        "smooth": L_smooth.item(),
        "div":    L_div.item(),
        "total":  total.item()
    }
    return total, components


# ══════════════════════════════════════════════════════════════════════════════
# COMPLETE RIFE-TIR MODEL
# ══════════════════════════════════════════════════════════════════════════════

class RIFE_TIR(nn.Module):
    """
    Full interpolation model for thermal IR satellite frames.

    Forward pass:
        1. IFNet estimates bilateral flow + occlusion mask
        2. Both frames are warped toward time t
        3. Coarse blend = weighted average of warped frames
        4. GridNet refines coarse blend → final output

    Args:
        t0      : (B, 1, H, W) normalised BT frame at time 0
        t1      : (B, 1, H, W) normalised BT frame at time 1
        t       : float or tensor, interpolation position in (0, 1)

    Returns:
        pred    : (B, 1, H, W) interpolated frame
        flow    : (B, 4, H, W) bilateral flow (for physics loss)
        mask    : (B, 1, H, W) occlusion mask
    """

    def __init__(self):
        super().__init__()
        self.ifnet   = IFNet()
        self.gridnet = GridNet(in_ch=8, base_ch=32)

    def forward(self, t0, t1, t=0.5):
        # ── Step 1: Flow estimation + warping ─────────────────────────────────
        warped_t0, warped_t1, flow, mask = self.ifnet(t0, t1, t)

        # ── Step 2: Coarse blend ──────────────────────────────────────────────
        # Weighted average using occlusion mask
        # mask ~ 1 means trust warped_t0 more, mask ~ 0 means trust warped_t1
        coarse = mask * warped_t0 + (1.0 - mask) * warped_t1

        # ── Step 3: GridNet refinement ────────────────────────────────────────
        pred = self.gridnet(warped_t0, warped_t1, mask, flow, coarse)

        return pred, flow, mask

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing RIFE-TIR complete model...")
    print("=" * 50)

    model = RIFE_TIR()
    print(f"IFNet params   : {sum(p.numel() for p in model.ifnet.parameters()):,}")
    print(f"GridNet params : {sum(p.numel() for p in model.gridnet.parameters()):,}")
    print(f"Total params   : {model.total_params():,}")
    print()

    # Simulate one training batch
    B, H, W = 2, 128, 128
    t0      = torch.rand(B, 1, H, W)
    t1      = torch.rand(B, 1, H, W)
    gt      = torch.rand(B, 1, H, W)   # ground truth middle frame
    t_param = torch.tensor([0.5, 0.5]) # interpolation position

    # Forward pass
    pred, flow, mask = model(t0, t1, t=0.5)
    print(f"pred shape  : {pred.shape}")
    print(f"flow shape  : {flow.shape}")
    print(f"mask shape  : {mask.shape}")
    print(f"pred range  : {pred.min():.3f} to {pred.max():.3f}")
    print()

    # Loss computation
    loss, components = physics_aware_loss(pred, flow, t0, t1, gt, t_param)
    print("Loss components:")
    for k, v in components.items():
        marker = " ← total" if k == "total" else ""
        print(f"  {k:<8}: {v:.6f}{marker}")
    print()

    # Backward pass check
    loss.backward()
    print("Backward pass : OK")
    print()
    print("RIFE-TIR model ready for training.")