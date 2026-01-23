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

#-----------------------------------------------------------------------------
# LSA Loss (CUDA)
#-----------------------------------------------------------------------------
def lsa_loss_cuda(truth, preds, colors):
    try:
        from torch_linear_assignment import batch_linear_assignment, assignment_to_indices
    except ModuleNotFoundError as e:
        print(
            "Please install torch-linear-assignment as"
            " `pip install torch-linear-assignment`"
            " to use LSA loss with Torch.")
        raise e

    total_loss = torch.tensor(0.0, device=preds.device)

    for color in [0, 1]:
        mask = (colors == color)

        counts = mask.sum(dim=1)
        unique_counts = torch.unique(counts)

        for k in unique_counts:
            k = k.item()
            if k == 0: continue

            batch_mask = (counts == k)

            sub_p = preds[batch_mask][mask[batch_mask]].view(-1, k, preds.shape[-1])
            sub_t = truth[batch_mask][mask[batch_mask]].view(-1, k, truth.shape[-1])

            # 1. Cost on [..., :3] (XYZ). This tensor HAS gradients.
            # dist_mat[b, i, j] = distance (i-th predicted , j-th truth) of b.
            dist_mat = ((sub_p.unsqueeze(2) - sub_t.unsqueeze(1)) ** 2).sum(dim=-1)
                # 21 x 257 x 257 say

            # 2. Solve Assignment
            assignment = batch_linear_assignment(dist_mat.detach()) # type: ignore
            _, col_ind = assignment_to_indices(assignment)          # type: ignore
                # 21 x 257

            # 3. Gather Loss directly using indices
            # We gather from the ORIGINAL 'dist_mat' to keep gradients flowing.
            matched_costs = dist_mat.gather(2, col_ind.unsqueeze(2).long()).squeeze(2)
                                                    #  | 21 x 257 x 1     -> 21 x 257
            total_loss += matched_costs.sum()

    return total_loss / truth.numel()

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
