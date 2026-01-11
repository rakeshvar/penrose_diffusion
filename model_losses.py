import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices


class PermutationLossScipy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, truth, preds, colors):
        batch_size = preds.shape[0]

        # 1. Compute Cost Matrix (squared euclidean distance)
        # Shape: (B, N, N) where Rows=Preds, Cols=Truth
        diff = preds.unsqueeze(2) - truth.unsqueeze(1)
        cost_matrix = (diff ** 2).sum(dim=-1)

        # 2. Detach for Scipy calculation
        cost_np = cost_matrix.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy() # Renamed to avoid confusion with loop var

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

        # 3. Gather Loss
        # Concatenate lists of arrays
        b_flat = np.concatenate(batch_ix)
        p_flat = np.concatenate(preds_ix)
        t_flat = np.concatenate(truth_ix)

        # Advanced indexing into the gradient-tracking tensor
        return cost_matrix[b_flat, p_flat, t_flat].sum() / truth.numel()

class PermutationOnlyScipy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, truth, preds, colors):
        batch_size = preds.shape[0]

        # 1. Compute Cost Matrix (squared euclidean distance)
        # Shape: (B, N, N) where Rows=Preds, Cols=Truth
        diff = preds.unsqueeze(2) - truth.unsqueeze(1)
        cost_matrix = (diff ** 2).sum(dim=-1)

        # 2. Detach for Scipy calculation
        cost_np = cost_matrix.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy() # Renamed to avoid confusion with loop var

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

        # 3. Gather Loss
        # Concatenate lists of arrays
        b_flat = np.concatenate(batch_ix)
        p_flat = np.concatenate(preds_ix)
        t_flat = np.concatenate(truth_ix)

        # Advanced indexing into the gradient-tracking tensor
        return b_flat, p_flat, t_flat

# !pip install -v torch-linear-assignment
class PermutationLossTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, truth, preds, colors):
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
                assignment = batch_linear_assignment(dist_mat.detach())
                _, col_ind = assignment_to_indices(assignment)
                    # 21 x 257

                # 3. Gather Loss directly using indices
                # We gather from the ORIGINAL 'dist_mat' to keep gradients flowing.
                matched_costs = dist_mat.gather(2, col_ind.unsqueeze(2).long()).squeeze(2)
                                                        #  | 21 x 257 x 1     -> 21 x 257
                total_loss += matched_costs.sum()

        return total_loss / truth.numel()
