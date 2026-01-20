import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

#-----------------------------------------------------------------------------
# Sin and Cos should be on a circle
#-----------------------------------------------------------------------------
def circle_loss(xysc_hat):
        sin_cos = xysc_hat[..., -2:]                              # B, N, 2
        return ((sin_cos.pow(2).sum(dim=-1) - 1.0) ** 2).mean()   # B, N

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


#-----------------------------------------------------------------------------
# Lattice Loss
#-----------------------------------------------------------------------------
def lattice_loss(xy_hat, s, t):
    """
    0 when all nearest neighbors are exactly s away.
    """
    _, N, D = xy_hat.shape
    if D > 2:
        xy_hat = xy_hat[..., :2]
    
    sq_dist = ((xy_hat.unsqueeze(2) - xy_hat.unsqueeze(1)) ** 2).sum(-1)
    dist = sq_dist ** .5
    dist += torch.eye(N, device=xy_hat.device).unsqueeze(0) * 1e6
    min_dist, min_idx = dist.min(-1)
    min_dist, min_idx = min_dist.squeeze(0), min_idx.squeeze(0)

    if isinstance(t, str):
        min_dist_rounded = min_dist.round(decimals=5)
        unique_vals, counts = torch.unique(min_dist_rounded, return_counts=True)

        if len(unique_vals) != 1:       
            print("Unique distances (6 decimal places):")
            for val, count in zip(unique_vals, counts):
                print(f"{val:.6f}: {count}")
            
            for i in range(N):
                if (min_dist[i] - s).abs() > 1e-4:
                    print(f"Point {i:2d}: {min_dist[i]:.5f} with point {min_idx[i]}")

            input("Press enter to continue...")
        
    return ((min_dist - s) ** 2).mean()