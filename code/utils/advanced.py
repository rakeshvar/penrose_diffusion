import math
from itertools import repeat
import torch

from code.polygons.hex.xya import HexGrid
from code.polygons.pen.xya import PenGrid
from code.polygons.hex.svg import save_svg as hex_save_svg
from code.polygons.pen.svg import save_svg as pen_save_svg

#------------------------------------------------------------------------------
# Save
#------------------------------------------------------------------------------
def xyac_to_svgs(xyac, symmetry, side, save_paths=repeat(None), print_ok=False):
    svgs = []
    xyac_np = xyac.detach().cpu().numpy()

    for sample, path in zip(xyac_np, save_paths):
        if symmetry == 6:
            grid = HexGrid(sample, side=side)
            svg = hex_save_svg(grid, path, print_ok=print_ok)
        else:
            grid = PenGrid(sample, from_np=True, side=side)
            svg = pen_save_svg(grid, path, print_ok=print_ok)

        svgs.append(svg)
    return svgs

#------------------------------------------------------------------------------
# Tensor Ops
#------------------------------------------------------------------------------

ANGLE_SCALE = math.sqrt(3.) / math.pi


def wrap_angle(angle):
    """Wrap angles in radians to [-π, π)."""
    return torch.remainder(angle + math.pi, 2. * math.pi) - math.pi


def xya_to_scaled(xya):
    """
    Convert (x, y, angle) to unit-variance (x, y, scaled_angle).

    A uniform angle on [-π, π) has variance π²/3, so multiplying by
    √3/π maps it to Uniform(-√3, √3), which has unit variance.
    """
    scaled = torch.stack(
        (xya[..., 0], xya[..., 1], wrap_angle(xya[..., 2]) * ANGLE_SCALE),
        dim=-1,
    )
    colors = xya[..., 3] if xya.shape[-1] > 3 else None
    return scaled, colors


def scaled_to_xyac(scaled_xya, colors=None):
    """Convert (x, y, scaled_angle) to radians, optionally appending color."""
    angle = wrap_angle(scaled_xya[..., 2] / ANGLE_SCALE)
    if colors is None:
        return torch.stack(
            (scaled_xya[..., 0], scaled_xya[..., 1], angle),
            dim=-1,
        )
    return torch.stack(
        (scaled_xya[..., 0], scaled_xya[..., 1], angle, colors),
        dim=-1,
    )


def sample_ot_noise(shape, device, dtype=torch.float32, generator=None):
    """Sample `(x, y, scaled_angle)` from N(0,I₂) × U(-√3,√3)."""
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"Expected a (B, N, 3) shape, got {shape}")
    xy = torch.randn(
        (*shape[:-1], 2),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    angle = (
        torch.rand(
            (*shape[:-1], 1),
            device=device,
            dtype=dtype,
            generator=generator,
        ) * 2. - 1.
    ) * math.sqrt(3.)
    return torch.cat((xy, angle), dim=-1)


def get_random_colors(symmetry, batch_size, num_tiles, device):
    prob = {6: 1/3, 5: (3-5**0.5)/2}[symmetry]
    unif = torch.rand((batch_size, num_tiles), device=device)
    colors = (unif < prob).long()
    return colors

def xysc_to_xyac(xysc, colors=None):
    """
    Convert (x, y, sinθ, cosθ) to (x, y, θ) or (x, y, θ, color).
    """
    x = xysc[..., 0]
    y = xysc[..., 1]
    s = xysc[..., 2]
    c = xysc[..., 3]
    angle = torch.arctan2(s, c)

    if colors is None:
        out = torch.stack([x, y, angle], dim=-1)
    else:
        out = torch.stack([x, y, angle, colors], dim=-1)

    return out


def xya_to_xysc(xya):
    """
    Convert (x, y, θ) or (x, y, θ, color) to (x, y, sinθ, cosθ).
    """
    x = xya[..., 0]
    y = xya[..., 1]
    a = xya[..., 2]

    s = torch.sin(a)
    c = torch.cos(a)
    xysc = torch.stack([x, y, s, c], dim=-1)

    colors = xya[..., 3] if xya.shape[-1] > 3 else None

    return xysc, colors


def pairwise_sq_dist(x, y, colors, σₑsq: float | torch.Tensor = 1.):
    """
    Used to caluclate raw logits:        ‖x₀ - ̂x₀‖² / 2σₑ²
    We use σₑ² here only to estimate BIG, to avoid overflow.
        exp(-100σₑ²/2σₑ²) = exp(-50) = 2e-22
    So colors are seperated by 10 units.
    """
    if isinstance(σₑsq, torch.Tensor):
        σₑsq = σₑsq.view(-1, 1, 1)

    x2 = (x ** 2).sum(dim=-1, keepdim=True)       # B, N, 1
    y2 = (y ** 2).sum(dim=-1, keepdim=True)       # B, M, 1
    xy = torch.bmm(x, y.transpose(1, 2))          # B, N, M
    sq_dist = (x2 + y2.transpose(1, 2) - 2 * xy).clamp_min(0.0)    # B, N, M

    BIG = 100. * σₑsq
    mismatch = (colors.unsqueeze(2) != colors.unsqueeze(1)).to(x.dtype)
    sq_dist = sq_dist + BIG * mismatch

    return sq_dist