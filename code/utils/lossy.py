import math
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

#---------------------------
# Sin and Cos should be on a circle
#---------------------------
def circle_loss_sincos(xysc_hat):
    sin_cos = xysc_hat[..., -2:]                 # B, N, 2
    norms = sin_cos.norm(dim=-1)                 # B, N
    target = torch.ones_like(norms)
    return F.mse_loss(norms, target)

#---------------------------
# All angles should be equal
#---------------------------
def equiangle_loss_sincos(xysc_hat, eps=1e-6):
    sin_cos = xysc_hat[..., -2:]                       # (B, N, 2)
    norms = sin_cos.norm(dim=-1, keepdim=True).clamp_min(eps)
    unit_sc = sin_cos / norms

    mean_sc = unit_sc.mean(dim=1)                       # (B, 2)
    mean_r = mean_sc.norm(dim=-1)                       # (B,)
    loss = 1. - mean_r.clamp(min=0.0, max=1.0)
    return loss.mean()

#---------------------------
# Lattice Loss
#---------------------------
def sq_dists(xy_hat):
    N = xy_hat.shape[1]
    xy = xy_hat[..., :2]
    sq_dist = ((xy.unsqueeze(2) - xy.unsqueeze(1)) ** 2).sum(-1)  # B, N, N
    dist_to_self = torch.eye(N, device=xy.device, dtype=sq_dist.dtype).unsqueeze(0) * 1e9
    sq_dist += dist_to_self
    return sq_dist

def _lattice_loss_quadratic(xy_hat, min_dist_to_nearest, max_dist_to_nearest, eps=1e-6):
    sq_dist = sq_dists(xy_hat)
    sq_dist_nn = sq_dist.min(-1)[0]
    dist_nn = torch.sqrt(sq_dist_nn + eps)

    r1 = dist_nn / min_dist_to_nearest
    r2 = dist_nn / max_dist_to_nearest
    loss = F.relu(1.0 - r1)**2 + F.relu(r2 - 1.0)**2
    return loss.mean()

#---------------------------
# Lattice Loss (NN attract + global log-repel)
#---------------------------
def _lattice_loss_logarithmic(xy_hat, min_dist_to_nearest, max_dist_to_nearest, eps=1e-6):
    sq_dist = sq_dists(xy_hat)
    dist = torch.sqrt(sq_dist+eps)

    eps2 = 1e-2 # To be safe on TPUs with bp16

    # nearest-neighbor attraction
    d_nn = dist.min(dim=-1)[0]
    r_nn = d_nn / max_dist_to_nearest
    gap_loss = r_nn - 1. - torch.log(r_nn + eps2)
    gap_mask = (r_nn > 1.0).to(gap_loss.dtype)
    gap_loss = (gap_loss * gap_mask).mean()

    # global logarithmic repulsion when overlapping
    r_ij = dist / min_dist_to_nearest
    lap_loss = r_ij - 1.0 - torch.log(r_ij + eps2)  # (B, N, N)
    lap_mask = (r_ij < 1.0).to(lap_loss.dtype)
    lap_loss = (lap_loss * lap_mask).mean()

    return (gap_loss + lap_loss) / 2.


def lattice_loss(symmetry, xy_hat, unit_side, algo="logarithmic"):
    if symmetry == 6:
        min_dist_to_nearest = max_dist_to_nearest = math.sqrt(3) * unit_side
    elif symmetry == 5:
        raise NotImplementedError
        min_dist_to_nearest = 0 * unit_side
        max_dist_to_nearest = 0 * unit_side
    else:
        raise ValueError(f"Unsupported symmetry: {symmetry} (must be 6 or 5)")

    if algo == "quadratic":
        return _lattice_loss_quadratic(xy_hat, min_dist_to_nearest, max_dist_to_nearest)
    elif algo == "logarithmic":
        return _lattice_loss_logarithmic(xy_hat, min_dist_to_nearest, max_dist_to_nearest)
    else:
        raise ValueError(f"Unsupported algo: {algo}")

#---------------------------
# Sinkhorn Soft-Permutation Calculation
#---------------------------
@torch.jit.script
def sinkhorn_permutation(log_K: torch.Tensor, n_iters: int = 7) -> torch.Tensor:
    """
    Batched Sinkhorn in log-space.
        log_K: [B, N, N] : - ||xi - xj||^2 / 2*σ^2
    Returns:
        P: [B, N, N] doubly-stochastic matrices
    """
    log_u = log_v = torch.zeros_like(log_K[..., 0])

    for _ in range(n_iters):
        # uᵢ = aᵢ /  ⟨Kᵢ. , v⟩
        log_u = -torch.logsumexp(log_K + log_v[:, None, :], dim=2)
        # vⱼ = bⱼ /  ⟨K.ⱼ , u⟩
        log_v = -torch.logsumexp(log_K + log_u[:, :, None], dim=1)

    # P = diag(u) K diag(v)
    log_P = log_K + log_u[:, :, None] + log_v[:, None, :]
    return torch.exp(log_P)

def sinkhorn_permutation_onestep(log_P: torch.Tensor) -> torch.Tensor:
    log_P = log_P - torch.logsumexp(log_P, dim=2, keepdim=True)  # rows
    log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)  # cols
    return torch.exp(log_P)

#---------------------------
# LSA Loss (Scipy)
#---------------------------
def lsa_ordering_scipy(cost_np, colors_np):
    """
    Computes optimal assignment indices using scipy's linear_sum_assignment.

    Returns:
        tuple: (batch_indices, truth_indices, pred_indices) as numpy arrays
    :param cost_np: Cost matrix
    :param colors_np: Subset of colors to match
    """
    batch_size = cost_np.shape[0]
    batch_ix, preds_ix, truth_ix = [], [], []

    for b in range(batch_size):
        for color in [0, 1]:
            select = np.where(colors_np[b] == color)[0]
            sub_cost = cost_np[b][np.ix_(select, select)]

            # linear_sum_assignment finds min cost matching
            # row_ind maps to 'select' (Predictions)
            # col_ind maps to 'select' (Truth)
            row_ind, col_ind = linear_sum_assignment(sub_cost)

            batch_ix.append(np.full(len(row_ind), b))
            preds_ix.append(select[row_ind])
            truth_ix.append(select[col_ind])

    # flatten lists of arrays
    b_flat = np.concatenate(batch_ix)
    t_flat = np.concatenate(truth_ix)
    p_flat = np.concatenate(preds_ix)

    return b_flat, t_flat, p_flat
