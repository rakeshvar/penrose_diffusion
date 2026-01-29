from itertools import repeat
import torch

from code.hex.base import HexGrid
from code.hex.svg import save_svg as hex_save_svg
from code.pen.base import PenGrid
from code.pen.svg import save_svg as pen_save_svg


#------------------------------------------------------------------------------
# Save
#------------------------------------------------------------------------------
def xyac_to_svgs(xyac, symmetry, side, save_paths=repeat(None)):
    svgs = []
    xyac_np = xyac.detach().cpu().numpy()

    for sample, path in zip(xyac_np, save_paths):
        if symmetry == 6:
            grid = HexGrid(sample, side=side)
            svg = hex_save_svg(grid, path)
        else:
            grid = PenGrid(sample, from_np=True, side=side)
            svg = pen_save_svg(grid, path)

        svgs.append(svg)
    return svgs

#------------------------------------------------------------------------------
# Tensor Ops
#------------------------------------------------------------------------------

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


def pairwise_sq_dist(x, y, colors, t):
    """
    x: (B, N, D)
    y: (B, M, D)
    colors: (B, N)
    t: (B,) ∈ [0, 1]
    """
    x2 = (x ** 2).sum(dim=-1, keepdim=True)       # B, N, 1
    y2 = (y ** 2).sum(dim=-1, keepdim=True)       # B, M, 1
    xy = torch.bmm(x, y.transpose(1, 2))          # B, N, M
    sq_dist = x2 + y2.transpose(1, 2) - 2 * xy    # B, N, M

    diff_color = colors[:, :, None] != colors[:, None, :]  # B, N, N
    BIG = (100.0 * t).view(-1, 1, 1)                        # B, 1, 1
    sq_dist = sq_dist + diff_color * BIG

    return sq_dist.clamp_min(0.0)
