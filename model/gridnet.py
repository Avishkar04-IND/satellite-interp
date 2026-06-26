"""
gridnet.py
GridNet refinement network — takes the two warped frames from IFNet
and fuses them into a single clean interpolated frame.
Suppresses warping artefacts and fills occlusion regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Basic blocks ──────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel,
                      stride=stride, padding=kernel//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class GridCell(nn.Module):
    """
    One cell in the GridNet grid.
    Receives input from left (same row) and top (row above, higher res).
    Produces output passed right and down.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip(x)


# ── GridNet ───────────────────────────────────────────────────────────────────

class GridNet(nn.Module):
    """
    GridNet refinement head — 3 rows x 6 columns of convolutional cells.

    Row 0 : full resolution  (H  x W  ) — 32 channels
    Row 1 : half resolution  (H/2x W/2) — 64 channels
    Row 2 : quarter res      (H/4x W/4) — 96 channels

    Columns 0-2 : encoding path  (lateral connections + downsampling)
    Columns 3-5 : decoding path  (lateral connections + upsampling)

    Input channels:
        warped_t0  : 1
        warped_t1  : 1
        mask       : 1
        flow       : 4
        coarse blend: 1
        ─────────────
        Total      : 9 channels
    """

    def __init__(self, in_ch=8, base_ch=32):
        super().__init__()
        ch0 = base_ch        # 32  — full resolution row
        ch1 = base_ch * 2    # 64  — half resolution row
        ch2 = base_ch * 3    # 96  — quarter resolution row

        # ── Encoding columns (0, 1, 2) ───────────────────────────────────────
        # Column 0
        self.enc_r0_c0 = GridCell(in_ch, ch0)
        self.enc_r1_c0 = GridCell(ch0,   ch1)   # + downsample from r0
        self.enc_r2_c0 = GridCell(ch1,   ch2)   # + downsample from r1

        # Column 1
        self.enc_r0_c1 = GridCell(ch0,       ch0)
        self.enc_r1_c1 = GridCell(ch1 + ch0, ch1)  # lateral + downsampled
        self.enc_r2_c1 = GridCell(ch2 + ch1, ch2)

        # Column 2
        self.enc_r0_c2 = GridCell(ch0,       ch0)
        self.enc_r1_c2 = GridCell(ch1 + ch0, ch1)
        self.enc_r2_c2 = GridCell(ch2 + ch1, ch2)

        # ── Decoding columns (3, 4, 5) ────────────────────────────────────────
        # Column 3
        self.dec_r2_c3 = GridCell(ch2,       ch2)
        self.dec_r1_c3 = GridCell(ch1 + ch2, ch1)  # lateral + upsampled
        self.dec_r0_c3 = GridCell(ch0 + ch1, ch0)

        # Column 4
        self.dec_r2_c4 = GridCell(ch2,       ch2)
        self.dec_r1_c4 = GridCell(ch1 + ch2, ch1)
        self.dec_r0_c4 = GridCell(ch0 + ch1, ch0)

        # Column 5
        self.dec_r2_c5 = GridCell(ch2,       ch2)
        self.dec_r1_c5 = GridCell(ch1 + ch2, ch1)
        self.dec_r0_c5 = GridCell(ch0 + ch1, ch0)

        # ── Output head ───────────────────────────────────────────────────────
        # Predicts a residual correction on top of the coarse blend
        self.output = nn.Sequential(
            ConvBNReLU(ch0, 16),
            nn.Conv2d(16, 1, 1),   # 1-channel BT residual
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _down(self, x):
        """Downsample 2x with average pooling."""
        return F.avg_pool2d(x, 2, 2)

    def _up(self, x, ref):
        """Upsample to match ref spatial size."""
        return F.interpolate(x, size=ref.shape[2:],
                             mode="bilinear", align_corners=False)

    def forward(self, warped_t0, warped_t1, mask, flow, coarse_blend):
        """
        Args:
            warped_t0    : (B, 1, H, W)  frame T0 warped toward time t
            warped_t1    : (B, 1, H, W)  frame T1 warped toward time t
            mask         : (B, 1, H, W)  occlusion mask from IFNet
            flow         : (B, 4, H, W)  bilateral flow fields
            coarse_blend : (B, 1, H, W)  simple weighted blend of warped frames
        Returns:
            refined      : (B, 1, H, W)  final interpolated frame
        """
        # Stack all inputs → 9 channels
        x = torch.cat([warped_t0, warped_t1, mask, flow, coarse_blend], dim=1)

        # ── Encoding path ─────────────────────────────────────────────────────
        # Column 0
        r0c0 = self.enc_r0_c0(x)
        r1c0 = self.enc_r1_c0(self._down(r0c0))
        r2c0 = self.enc_r2_c0(self._down(r1c0))

        # Column 1
        r0c1 = self.enc_r0_c1(r0c0)
        r1c1 = self.enc_r1_c1(torch.cat([r1c0, self._down(r0c1)], dim=1))
        r2c1 = self.enc_r2_c1(torch.cat([r2c0, self._down(r1c1)], dim=1))

        # Column 2
        r0c2 = self.enc_r0_c2(r0c1)
        r1c2 = self.enc_r1_c2(torch.cat([r1c1, self._down(r0c2)], dim=1))
        r2c2 = self.enc_r2_c2(torch.cat([r2c1, self._down(r1c2)], dim=1))

        # ── Decoding path ─────────────────────────────────────────────────────
        # Column 3
        r2c3 = self.dec_r2_c3(r2c2)
        r1c3 = self.dec_r1_c3(torch.cat([r1c2, self._up(r2c3, r1c2)], dim=1))
        r0c3 = self.dec_r0_c3(torch.cat([r0c2, self._up(r1c3, r0c2)], dim=1))

        # Column 4
        r2c4 = self.dec_r2_c4(r2c3)
        r1c4 = self.dec_r1_c4(torch.cat([r1c3, self._up(r2c4, r1c3)], dim=1))
        r0c4 = self.dec_r0_c4(torch.cat([r0c3, self._up(r1c4, r0c3)], dim=1))

        # Column 5
        r2c5 = self.dec_r2_c5(r2c4)
        r1c5 = self.dec_r1_c5(torch.cat([r1c4, self._up(r2c5, r1c4)], dim=1))
        r0c5 = self.dec_r0_c5(torch.cat([r0c4, self._up(r1c5, r0c4)], dim=1))

        # ── Output: residual on coarse blend ──────────────────────────────────
        residual = self.output(r0c5)
        refined  = coarse_blend + residual

        # Clamp to valid normalised BT range [0, 1]
        refined  = torch.clamp(refined, 0.0, 1.0)

        return refined


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing GridNet...")
    model = GridNet(in_ch=8, base_ch=32)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total:,}")

    B, H, W = 2, 128, 128
    warped_t0    = torch.rand(B, 1, H, W)
    warped_t1    = torch.rand(B, 1, H, W)
    mask         = torch.rand(B, 1, H, W)
    flow         = torch.rand(B, 4, H, W)
    coarse_blend = torch.rand(B, 1, H, W)

    refined = model(warped_t0, warped_t1, mask, flow, coarse_blend)

    print(f"Input  : warped_t0{list(warped_t0.shape)} + "
          f"warped_t1{list(warped_t1.shape)} + "
          f"mask{list(mask.shape)} + "
          f"flow{list(flow.shape)} + "
          f"blend{list(coarse_blend.shape)}")
    print(f"Output : {refined.shape}")
    print(f"Range  : {refined.min():.3f} to {refined.max():.3f}")
    print("\nGridNet OK")