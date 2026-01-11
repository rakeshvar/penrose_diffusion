import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices

class PermutationLossScipy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, truth):
        batch_size, _, _ = pred.shape

        # 1. Compute Cost Matrix on [..., :3] (XYZ) immediately
        # This tensor HAS gradients from pred.
        diff = pred[..., :3].unsqueeze(2) - truth[..., :3].unsqueeze(1)
        cost_matrix = (diff ** 2).sum(dim=-1)

        # Detach only for the numpy solver logic
        cost_np = cost_matrix.detach().cpu().numpy()
        pred_c = pred[..., 3].detach().cpu().numpy().astype(int)
        truth_c = truth[..., 3].detach().cpu().numpy().astype(int)

        b_idxs, p_idxs, t_idxs = [], [], []

        for b in range(batch_size):
            for color in [0, 1]:
                p_idx = np.where(pred_c[b] == color)[0]
                t_idx = np.where(truth_c[b] == color)[0]

                if len(p_idx) == 0: continue

                # Solve sub-problem
                sub_cost = cost_np[b][np.ix_(p_idx, t_idx)]
                r_ind, c_ind = linear_sum_assignment(sub_cost)

                b_idxs.append(np.full(len(r_ind), b))
                p_idxs.append(p_idx[r_ind])
                t_idxs.append(t_idx[c_ind])

        if not b_idxs:
            return (pred * 0).sum()

        # 2. Gather Loss directly
        # We index into the ORIGINAL 'cost_matrix' (which has grads).
        # This preserves the gradient chain back to 'pred'.
        b_flat = np.concatenate(b_idxs)
        p_flat = np.concatenate(p_idxs)
        t_flat = np.concatenate(t_idxs)

        return cost_matrix[b_flat, p_flat, t_flat].sum()

# !pip install -v torch-linear-assignment
class PermutationLossTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, truth):
        total_loss = torch.tensor(0.0, device=pred.device)

        for color in [0, 1]:
            mask_p = (pred[..., 3] == color)
            mask_t = (truth[..., 3] == color)

            counts = mask_p.sum(dim=1)
            unique_counts = torch.unique(counts)

            for k in unique_counts:
                k = k.item()
                if k == 0: continue

                batch_mask = (counts == k)

                sub_p = pred[batch_mask][mask_p[batch_mask]].view(-1, k, 4)
                sub_t = truth[batch_mask][mask_t[batch_mask]].view(-1, k, 4)

                # 1. Cost on [..., :3] (XYZ). This tensor HAS gradients.
                dist_mat = ((sub_p[..., :3].unsqueeze(2) - sub_t[..., :3].unsqueeze(1)) ** 2).sum(dim=-1)

                # 2. Solve Assignment
                # FIX: Detach dist_mat for assignment solver.
                # We only need integer indices here, which are not differentiable.
                assignment = batch_linear_assignment(dist_mat.detach())
                _, col_ind = assignment_to_indices(assignment)

                # 3. Gather Loss directly using indices
                # We gather from the ORIGINAL 'dist_mat' to keep gradients flowing.
                matched_costs = dist_mat.gather(2, col_ind.unsqueeze(2).long()).squeeze(2)
                total_loss += matched_costs.sum()

        return total_loss
