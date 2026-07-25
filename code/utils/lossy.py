import math
import os
from concurrent.futures import ThreadPoolExecutor

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
    lap_loss = (lap_loss * lap_mask).sum(dim=-1).mean()

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
def _solve_unconstrained_lsa(cost):
    """Return column index assigned to every row of one square cost matrix."""
    row_ind, col_ind = linear_sum_assignment(cost)
    permutation = np.empty(cost.shape[0], dtype=np.int64)
    permutation[row_ind] = col_ind
    return permutation


class ScipyBatchedLSA:
    """Persistent threaded solver for independent dense assignment problems."""
    def __init__(self, max_workers=None):
        available = os.cpu_count() or 1
        self.max_workers = max(1, min(int(max_workers or available), available))
        self._executor = (
            ThreadPoolExecutor(max_workers=self.max_workers)
            if self.max_workers > 1
            else None
        )
        self._closed = False

    def solve_numpy(self, cost):
        """Solve a `(B, N, N)` NumPy cost array and return `(B, N)` indices."""
        if self._closed:
            raise RuntimeError("ScipyBatchedLSA is closed")
        if cost.ndim != 3 or cost.shape[1] != cost.shape[2]:
            raise ValueError(f"Expected square batched costs, got shape {cost.shape}")

        if self._executor is None:
            permutations = [_solve_unconstrained_lsa(matrix) for matrix in cost]
        else:
            permutations = list(self._executor.map(_solve_unconstrained_lsa, cost))
        return np.stack(permutations)

    def solve(self, cost):
        """Synchronously solve a tensor cost batch and return indices on its device."""
        permutation = self.solve_numpy(cost.detach().cpu().numpy())
        return torch.from_numpy(permutation).to(cost.device, non_blocking=True)

    def close(self):
        if not self._closed:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def ot_cost_matrix(x0, noise):
    """Squared Euclidean tile costs with shape `(B, N, N)`."""
    if x0.shape != noise.shape or x0.ndim != 3:
        raise ValueError(f"Expected matching (B, N, D) tensors, got {x0.shape} and {noise.shape}")
    return torch.cdist(x0, noise).square()


def gather_by_permutation(values, permutation):
    """Gather `(B, N, D)` values according to a `(B, N)` permutation."""
    if values.ndim != 3 or permutation.shape != values.shape[:2]:
        raise ValueError(
            f"Expected values (B, N, D) and permutation (B, N), got "
            f"{values.shape} and {permutation.shape}"
        )
    indices = permutation.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return values.gather(1, indices)


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


#---------------------------
# Soft Assignment & LSA Abstractions
#---------------------------
def soft_assignment_matrix(x_query, x_key, colors, variance, method='softmax'):
    """
    Computes soft permutation matrix between query and key point sets.
    """
    from code.utils.advanced import pairwise_sq_dist
    sq_dist = pairwise_sq_dist(x_query, x_key, colors, variance)
    logits = -sq_dist / (2.0 * variance)
    
    if method == 'sinkhorn':
        return sinkhorn_permutation(logits)
    elif method == 'softmax':
        return torch.softmax(logits, dim=-1)
    else:
        raise ValueError(f"Unknown assignment method: {method}")


def get_lsa_indices(x_query, x_key, colors):
    """
    Computes optimal bipartite matching between query and key point sets.
    """
    diff = x_query.unsqueeze(2) - x_key.unsqueeze(1)
    cost_matrix = (diff ** 2).sum(dim=-1)
    cost_np = cost_matrix.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()

    bi, ti, pi = lsa_ordering_scipy(cost_np, colors_np)
    device = x_query.device
    bi = torch.from_numpy(bi).to(device, non_blocking=True)
    ti = torch.from_numpy(ti).to(device, non_blocking=True)
    pi = torch.from_numpy(pi).to(device, non_blocking=True)
    return bi, ti, pi
