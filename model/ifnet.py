"""
ifnet.py
Implicit Flow Network — estimates bilateral optical flow between
two thermal IR frames without explicitly computing traditional optical flow.
Adapted from RIFE for single-channel (grayscale) satellite TIR input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Basic building blocks ─────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BatchNorm → PReLU"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Two ConvBlocks with a residual skip connection."""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
        self.relu = nn.PReLU(ch)

    def forward(self, x):
        return self.relu(x + self.block(x))


# ── Warping utility ───────────────────────────────────────────────────────────

def warp(frame, flow):
    """
    Backward warp a frame using a flow field.
    frame : (B, C, H, W)
    flow  : (B, 2, H, W) — (u, v) displacement in pixels
    Returns warped frame of same shape.
    """
    B, C, H, W = frame.shape

    # Build base sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=frame.device),
        torch.arange(W, dtype=torch.float32, device=frame.device),
        indexing="ij"
    )
    grid = torch.stack([grid_x, grid_y], dim=0)   # (2, H, W)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1) # (B, 2, H, W)

    # Add flow displacement
    new_grid = grid + flow

    # Normalise to [-1, 1] for grid_sample
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0

    # Permute to (B, H, W, 2) as required by grid_sample
    new_grid = new_grid.permute(0, 2, 3, 1)

    return F.grid_sample(
        frame, new_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )


# ── Encoder (feature pyramid) ─────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    4-scale feature pyramid encoder.
    Takes a single TIR frame (1 channel) and produces
    multi-scale feature maps for flow estimation.
    """
    def __init__(self):
        super().__init__()
        # Scale 1 — full resolution
        self.layer1 = nn.Sequential(
            ConvBlock(1,  16, stride=1),
            ResBlock(16)
        )
        # Scale 2 — 1/2 resolution
        self.layer2 = nn.Sequential(
            ConvBlock(16, 32, stride=2),
            ResBlock(32)
        )
        # Scale 3 — 1/4 resolution
        self.layer3 = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ResBlock(64)
        )
        # Scale 4 — 1/8 resolution
        self.layer4 = nn.Sequential(
            ConvBlock(64, 96, stride=2),
            ResBlock(96)
        )

    def forward(self, x):
        f1 = self.layer1(x)   # (B, 16, H,   W  )
        f2 = self.layer2(f1)  # (B, 32, H/2, W/2)
        f3 = self.layer3(f2)  # (B, 64, H/4, W/4)
        f4 = self.layer4(f3)  # (B, 96, H/8, W/8)
        return f1, f2, f3, f4


# ── Flow decoder (coarse to fine) ────────────────────────────────────────────

class FlowDecoder(nn.Module):
    """
    Estimates bilateral flow at one pyramid scale.
    Input: concatenated features from T0 and T1 + upsampled flow from coarser scale.
    Output: refined flow (4 channels: u01, v01, u10, v10) + occlusion mask.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, 96),
            ConvBlock(96,    64),
            ConvBlock(64,    32),
            # Output: 4 flow channels + 1 occlusion mask
            nn.Conv2d(32, 5, 3, padding=1)
        )

    def forward(self, x):
        out  = self.net(x)
        flow = out[:, :4, :, :]          # (B, 4, H, W)
        mask = torch.sigmoid(out[:, 4:]) # (B, 1, H, W) occlusion mask
        return flow, mask


# ── IFNet — full implicit flow estimator ─────────────────────────────────────

class IFNet(nn.Module):
    """
    Implicit Flow Network for thermal IR frame interpolation.

    Given two TIR frames (T0, T1) and a time parameter t,
    estimates bilateral flow fields F_01 and F_10, warps both
    frames toward time t, and blends using an occlusion mask.

    Input : T0 (B,1,H,W), T1 (B,1,H,W), t scalar
    Output: warped_t0 (B,1,H,W), warped_t1 (B,1,H,W),
            flow (B,4,H,W), mask (B,1,H,W)
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        # Flow decoders for each scale (coarse to fine)
        # in_ch = feat_t0 + feat_t1 + flow_upsample + t_plane
        self.decoder4 = FlowDecoder(96*2 + 0  + 1)  # coarsest, no prior flow
        self.decoder3 = FlowDecoder(64*2 + 5  + 1)
        self.decoder2 = FlowDecoder(32*2 + 5  + 1)
        self.decoder1 = FlowDecoder(16*2 + 5  + 1)

    def _t_plane(self, t, B, H, W, device):
        """Create a constant plane filled with value t, shape (B,1,H,W)."""
        return torch.full((B, 1, H, W), t, dtype=torch.float32, device=device)

    def forward(self, t0, t1, t=0.5):
        B, _, H, W = t0.shape
        device = t0.device

        # Encode both frames
        f0_1, f0_2, f0_3, f0_4 = self.encoder(t0)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(t1)

        # ── Scale 4 (coarsest, 1/8 resolution) ───────────────────────────────
        h4, w4 = H // 8, W // 8
        t4     = self._t_plane(t, B, h4, w4, device)
        inp4   = torch.cat([f0_4, f1_4, t4], dim=1)
        flow4, mask4 = self.decoder4(inp4)

        # ── Scale 3 (1/4 resolution) ──────────────────────────────────────────
        flow4_up = F.interpolate(flow4, scale_factor=2, mode="bilinear", align_corners=False) * 2
        mask4_up = F.interpolate(mask4, scale_factor=2, mode="bilinear", align_corners=False)
        h3, w3   = H // 4, W // 4
        t3       = self._t_plane(t, B, h3, w3, device)
        prior3   = torch.cat([flow4_up, mask4_up], dim=1)  # 5 channels
        inp3     = torch.cat([f0_3, f1_3, prior3, t3], dim=1)
        flow3, mask3 = self.decoder3(inp3)
        flow3    = flow3 + flow4_up  # residual refinement

        # ── Scale 2 (1/2 resolution) ──────────────────────────────────────────
        flow3_up = F.interpolate(flow3, scale_factor=2, mode="bilinear", align_corners=False) * 2
        mask3_up = F.interpolate(mask3, scale_factor=2, mode="bilinear", align_corners=False)
        h2, w2   = H // 2, W // 2
        t2       = self._t_plane(t, B, h2, w2, device)
        prior2   = torch.cat([flow3_up, mask3_up], dim=1)
        inp2     = torch.cat([f0_2, f1_2, prior2, t2], dim=1)
        flow2, mask2 = self.decoder2(inp2)
        flow2    = flow2 + flow3_up

        # ── Scale 1 (full resolution) ─────────────────────────────────────────
        flow2_up = F.interpolate(flow2, scale_factor=2, mode="bilinear", align_corners=False) * 2
        mask2_up = F.interpolate(mask2, scale_factor=2, mode="bilinear", align_corners=False)
        t1_plane = self._t_plane(t, B, H, W, device)
        prior1   = torch.cat([flow2_up, mask2_up], dim=1)
        inp1     = torch.cat([f0_1, f1_1, prior1, t1_plane], dim=1)
        flow1, mask1 = self.decoder1(inp1)
        flow1    = flow1 + flow2_up   # final full-res flow

        # ── Warp both frames toward time t ────────────────────────────────────
        # flow1 channels: (u_01, v_01, u_10, v_10)
        # F_01: flow from T0 toward T1, scaled by t
        # F_10: flow from T1 toward T0, scaled by (1-t)
        f_01 = flow1[:, 0:2] *  t         # T0 → interpolated time
        f_10 = flow1[:, 2:4] * (1.0 - t)  # T1 → interpolated time

        warped_t0 = warp(t0, f_01)
        warped_t1 = warp(t1, f_10)

        return warped_t0, warped_t1, flow1, mask1


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing IFNet...")
    model = IFNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total_params:,}")

    # Simulate one batch
    B, H, W = 2, 128, 128
    t0 = torch.rand(B, 1, H, W)
    t1 = torch.rand(B, 1, H, W)

    warped_t0, warped_t1, flow, mask = model(t0, t1, t=0.5)

    print(f"warped_t0  : {warped_t0.shape}")
    print(f"warped_t1  : {warped_t1.shape}")
    print(f"flow       : {flow.shape}")
    print(f"mask       : {mask.shape}")
    print(f"flow range : {flow.min():.3f} to {flow.max():.3f}")
    print(f"mask range : {mask.min():.3f} to {mask.max():.3f}")
    print("\nIFNet OK")