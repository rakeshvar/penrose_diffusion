import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

#-----------------------------------------------------------------------------
# Sin and Cos should be on a circle, they all should be equal for hex girds
#-----------------------------------------------------------------------------
def circle_loss(xysc_hat):
    sin_cos = xysc_hat[..., -2:]                 # B, N, 2
    norms = sin_cos.norm(dim=-1)                 # B, N
    target = torch.ones_like(norms)
    return F.mse_loss(norms, target)

def equal_angle_loss_var(xysc_hat, eps=1e-8):
    sin_cos = xysc_hat[..., -2:]                      # (B, N, 2)
    norms = sin_cos.norm(dim=-1, keepdim=True).clamp_min(eps)
    unit = sin_cos / norms                            # (B, N, 2)
    var = unit.var(dim=1, unbiased=False)             # (B, 2) stable scale (div by N)
    return var.mean()

def equal_angle_loss_circular(xysc_hat, eps=1e-6):
    sin_cos = xysc_hat[..., -2:]                       # (B, N, 2)
    norms = sin_cos.norm(dim=-1, keepdim=True).clamp_min(eps)
    unit = sin_cos / norms                             # (B, N, 2)

    mean_vec = unit.mean(dim=1)                        # (B, 2)
    R = mean_vec.norm(dim=-1)                          # (B,)

    # avoid dividing by exactly zero; when R is tiny, unit_m is arbitrary but stable
    invR = 1.0 / (R.clamp_min(eps))                    # (B,)
    unit_m = mean_vec * invR.unsqueeze(-1)             # (B, 2)

    # resultant length loss
    circ_var = 1.0 - R.clamp(min=0.0, max=1.0)         # (B,) in [0,1]
    return circ_var.mean()

#-----------------------------------------------------------------------------
# Lattice Loss
#-----------------------------------------------------------------------------
def lattice_loss(xy_hat, unit_side, symmetry):
    """
    0 when all nearest neighbors are exactly s units away.
    """
    side2nearest = {6: 3**0.5}[symmetry]
    to_nearest = unit_side * side2nearest

    B, N, D = xy_hat.shape
    xy_hat = xy_hat[..., :2]

    sq_dist = ((xy_hat.unsqueeze(2) - xy_hat.unsqueeze(1)) ** 2).sum(-1)  # B, N, N
    dist_to_self = torch.eye(N, device=xy_hat.device, dtype=sq_dist.dtype).unsqueeze(0) * 1e6
    sq_dist += dist_to_self
    min_sq_dist = sq_dist.min(-1)[0]                    # B, N
    min_dist = torch.sqrt(min_sq_dist)

    target = torch.full_like(min_dist, fill_value=to_nearest)
    return F.mse_loss(min_dist, target) / to_nearest**2

def geodesic_loss(xysc_0, xysc_0_hat):
    sc_0 = xysc_0[..., -2:]
    sc_0_hat = xysc_0_hat[..., -2:]

    u_0 = sc_0 / (sc_0.norm(dim=-1, keepdim=True) + 1e-6)
    u_0_hat = sc_0_hat / (sc_0_hat.norm(dim=-1, keepdim=True) + 1e-6)

    # cosine similarity
    cos_dtheta = (u_0 * u_0_hat).sum(dim=-1)  # (B, N)

    # geodesic loss on S¹
    loss_angle = (1.0 - cos_dtheta).mean()

def tangent_space_score_loss(xysc_t, noise, noise_hat):
    # angular components
    sincos_t = xysc_t[..., 2:]          # (B, N, 2)
    eps_true = noise[..., 2:]
    eps_hat = noise_hat[..., 2:]

    # unit direction (safe)
    u = sincos_t / (sincos_t.norm(dim=-1, keepdim=True) + 1e-6)

    # project noise to tangent space
    def project_tangent(eps, u):
        return eps - (eps * u).sum(dim=-1, keepdim=True) * u

    eps_true_tan = project_tangent(eps_true, u)
    eps_hat_tan = project_tangent(eps_hat, u)

    # final loss (add x,y MSE normally)
    loss_angle = F.mse_loss(eps_hat_tan, eps_true_tan)

#-----------------------------------------------------------------------------
# Sinkhorn Soft-Permutation Calculation
#-----------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------
# LSA Loss (Scipy)
#-----------------------------------------------------------------------------
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
