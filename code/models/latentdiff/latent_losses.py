import math
import torch
import torch.nn.functional as F
from ...utils.lossy import soft_assignment_matrix, get_lsa_indices
from ...utils.registry import Registry

loss_registry = Registry(name="LatentLoss")
register_loss = loss_registry.register

def t_from_s(s=.18, R=100):                 # .18 is unit_side of polygon
    d = math.sqrt(3) * s                    # distance to nearest neighbor
    t = d/math.sqrt(2 * math.log(R))        # first one is R factor more than second
    return t

# ------------------------------------------------------------
# Reconstruction losses
# ------------------------------------------------------------

@register_loss('spl', 'samp', 'sample', 'sampleloss')
def sample_loss(x, y, colors):
    return F.mse_loss(x, y)


@register_loss('cfl', 'cham', 'chamfer')
def chamfer_loss(x, y, colors):
    from code.utils.advanced import pairwise_sq_dist
    sq_dist = pairwise_sq_dist(x, y, colors)                # B, N, N
    loss_xy = sq_dist.min(dim=2).values.mean()
    loss_yx = sq_dist.min(dim=1).values.mean()
    return (loss_xy + loss_yx)/2.


@register_loss('shl', 'sink', 'sinkhorn')
def sinkhorn_loss(x, y, colors):
    variance = t_from_s()**2
    P = soft_assignment_matrix(x, y, colors, variance, 'sinkhorn')
    y_post = torch.bmm(P, y)
    return F.mse_loss(x, y_post)


@register_loss('pil', 'pinv', 'perminv')
def pinv_loss(x, y, colors):
    variance = t_from_s()**2
    P = soft_assignment_matrix(x, y, colors, variance, 'softmax')
    y_post = torch.bmm(P, y)
    return F.mse_loss(x, y_post)


@register_loss('lsl', 'lsas', 'lsaserial', 'lpl', 'lsap', 'lsaparallel')
def lsa_loss(x, y, colors):
    bi, ti, pi = get_lsa_indices(y, x, colors)
    return F.mse_loss(x[bi, ti], y[bi, pi])
