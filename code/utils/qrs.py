import numpy as np
from .basic import TablePrinter


def qrs_npz_stats(npz_name):
    print("\nFile: ", npz_name)

    with np.load(npz_name) as data:
        qr = data['qr']
        q = qr[..., 0]
        r = qr[..., 1]
        s = -r - q

    tp = TablePrinter(7, 11)
    tp.top_line()
    tp.line("Variable", "Global/Ind.", "min", "mean", "max", "std", "range")

    def stats(v, name):
        tp.mid_line()
        tp.line(name, "global", np.min(v), np.mean(v), np.max(v), np.std(v), np.max(v) - np.min(v))
        tp.line(name, "indiv. avg",
                np.min(v, axis=-1).mean(), np.mean(v, axis=-1).mean(),
                np.max(v, axis=-1).mean(), np.std(v, axis=-1).mean(),
                (np.max(v, axis=-1) - np.min(v, axis=-1)).mean())
        tp.line(name, "indiv. max",
                np.min(v, axis=-1).max(), np.mean(v, axis=-1).max(),
                np.max(v, axis=-1).max(), np.std(v, axis=-1).max(),
                (np.max(v, axis=-1) - np.min(v, axis=-1)).max())
        tp.line(name, "indiv. min",
                np.min(v, axis=-1).min(), np.mean(v, axis=-1).min(),
                np.max(v, axis=-1).min(), np.std(v, axis=-1).min(),
                (np.max(v, axis=-1) - np.min(v, axis=-1)).min())


    stats(q, "q")
    stats(r, "r")
    stats(s, "s")
    tp.bot_line()


import math
import torch


def xya_to_qr(xya, side):
    # 1. Normalize Rotation
    grid_angle = xya[..., 2].mean(dim=1, keepdim=True) # (B, 1)
    cosθ = torch.cos(-grid_angle)
    sinθ = torch.sin(-grid_angle)
    x = xya[..., 0]
    y = xya[..., 1]        
    x_rot = x * cosθ - y * sinθ
    y_rot = x * sinθ + y * cosθ
    
    # Move (x, y) so that the center most tile is origin
    # 1. Find the geometric centroid of the cluster
    centroid_x = x_rot.mean(dim=1, keepdim=True)
    centroid_y = y_rot.mean(dim=1, keepdim=True)
    
    # 2. Find the index of the tile closest to the centroid
    dists = (x_rot - centroid_x).pow(2) + (y_rot - centroid_y).pow(2)
    center_idx = torch.argmin(dists, dim=1) # (B,)
    
    # 3. Gather the coordinates of that center tile
    batch_idx = torch.arange(xya.shape[0], device=xya.device)
    offset_x = x_rot[batch_idx, center_idx].unsqueeze(1)
    offset_y = y_rot[batch_idx, center_idx].unsqueeze(1)
    
    # 4. Shift all points relative to the center tile
    x_rot = x_rot - offset_x
    y_rot = y_rot - offset_y

    r_float = (2.0 / 3.0) * y_rot / side
    q_float = (x_rot / (side * math.sqrt(3.0))) - (r_float / 2.0)
    q = torch.round(q_float).long()
    r = torch.round(r_float).long()
    return q, r

def qr_to_xya(q, r, side, rotateθ=None):
    x = side * math.sqrt(3) * (q + r / 2.0)
    y = side * 1.5 * r
    
    if rotateθ is not None:
        # rotateθ: (B, 1)
        cosθ = torch.cos(rotateθ)
        sinθ = torch.sin(rotateθ)
        x_new = x * cosθ - y * sinθ
        y_new = x * sinθ + y * cosθ
        angle = rotateθ.expand_as(x)
        x, y = x_new, y_new
    else:
        angle = torch.zeros_like(x)

    return torch.stack([x, y, angle], dim=-1)


def get_colors(q, r):
    a = torch.stack([q, r, -q - r], dim=-1).abs()
    mx = a.max(dim=-1)[0]
    mn = a.min(dim=-1)[0]
    return ((mx + mn) % 3 == 0).long()

def spiral_sort(q, r):
    """
    Sorts the tiles to ensure a deterministic sequence order.
    Sorts by degree (shells), then circularly by angle.
    """
    s = -q - r
    abs_q, abs_r, abs_s = q.abs(), r.abs(), s.abs()
    degree = torch.max(torch.stack([abs_q, abs_r, abs_s], dim=-1), dim=-1).values

    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    angle = torch.atan2(y, x) # Result is in (-pi, pi]
    
    key = degree.float() * 10. + angle 
    sort_idx = torch.argsort(key, dim=1)
    
    q_sorted = torch.gather(q, 1, sort_idx)
    r_sorted = torch.gather(r, 1, sort_idx)
    return q_sorted, r_sorted

def spiral_sort_qrs(qr):
    """
    Sorts tiles by distance from center (shell) then by angle.
    CRITICAL: Also re-centers the cluster to (0,0) so vocab stays small.
    """
    print("Sorting QRs... ", qr.shape, qr.dtype)
    M, N, D = qr.shape
    
    q = qr[..., 0].float()
    r = qr[..., 1].float()
    
    # Mean-center
    q_center = q.mean(dim=1, keepdim=True).round()
    r_center = r.mean(dim=1, keepdim=True).round()
    q_centered = q - q_center
    r_centered = r - r_center
    s_centered = -q_centered - r_centered 
    qr_centered = torch.stack([q_centered, r_centered], dim=2).long()
    
    # Sort 
    # Key A: Radius = max(|q|, |r|, |s|)
    radius = torch.max(torch.max(q_centered.abs(), r_centered.abs()), s_centered.abs())
    # Key B: Angle
    x, y = math.sqrt(3) * (q_centered + 0.5 * r_centered), 1.5 * r_centered
    angle = torch.atan2(y, x)               # (-pi, pi]
    sort_key = 10 * radius - angle          # radius, then clockwise
    perm = torch.argsort(sort_key, dim=1) # (M, N)
  
    # 3D gather: (M, N) -> (M, N, D)
    perm_expanded = perm.unsqueeze(-1).expand(-1, -1, D)
    data_sorted = torch.gather(qr_centered, 1, perm_expanded)
    return data_sorted