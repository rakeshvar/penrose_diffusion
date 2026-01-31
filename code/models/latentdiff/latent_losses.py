import torch
import torch.nn.functional as F
from code.utils.advanced import pairwise_sq_dist
from code.utils.lossy import sinkhorn_permutation
from ...utils.registry import Registry

loss_registry = Registry(name="LatentLoss")
register_loss = loss_registry.register

# ------------------------------------------------------------
# Reconstruction losses
# ------------------------------------------------------------

@register_loss('samp', 'spl', 'sample', 'sampleloss')
def sample_loss(x, y, colors):
    return F.mse_loss(x, y)

@register_loss('cham', 'chamfer', 'cfl')
def chamfer_loss(x, y, colors):
    sq_dist = pairwise_sq_dist(x, y, colors)                # B, N, N
    loss_xy = sq_dist.min(dim=2).values.mean()
    loss_yx = sq_dist.min(dim=1).values.mean()
    return (loss_xy + loss_yx)/2.

@register_loss('sink', 'sinkhorn', 'shl')
def sinkhorn_loss(x, y, colors):
    sq_dist = pairwise_sq_dist(x, y, colors)         # B, N, N   
    logits = -sq_dist/(2*.15**2)                     # .15 is unit_side of polygon
    P = sinkhorn_permutation(logits)
    y_post = torch.bmm(P, y)
    return F.mse_loss(x, y_post)

@register_loss('pinv', 'pil', 'perminv')
def pinv_loss(x, y, colors):
    sq_dist = pairwise_sq_dist(x, y, colors)         # B, N, N   
    logits = -sq_dist/(2*.15**2)                     # .15 is unit_side of polygon
    P = torch.softmax(logits, dim=-1)
    y_post = torch.bmm(P, y)
    return F.mse_loss(x, y_post)

