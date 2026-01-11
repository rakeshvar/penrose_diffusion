import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices
from utils import pairwise_compare as compare
from utils import linear_compare as compare

# ==========================================
# Only use XY to calculate distance
# ==========================================
class ScipyPermutationLossXY(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, truth):
        batch_size, num_tokens, _ = pred.shape
        device = pred.device

        # Cost Matrix on [..., :2] (XY only)
        pred_xy = pred[..., :2]
        truth_xy = truth[..., :2]
        sq_diff = (pred_xy.unsqueeze(2) - truth_xy.unsqueeze(1)) ** 2
        distmat = sq_diff.sum(dim=-1)

        distmat_np = distmat.detach().cpu().numpy()
        pred_color_np = pred[..., 3].detach().cpu().numpy().astype(int)
        truth_color_np = truth[..., 3].detach().cpu().numpy().astype(int)

        final_perm_indices = []

        for b in range(batch_size):
            batch_col_indices = np.empty(num_tokens, dtype=np.int64)
            p_colors = pred_color_np[b]
            t_colors = truth_color_np[b]

            for color in [0, 1]:
                p_idx = np.where(p_colors == color)[0]
                t_idx = np.where(t_colors == color)[0]

                if len(p_idx) == 0: continue

                # Extract sub-square matrix
                # np.ix_ creates the meshgrid for indexing
                sub_dist = distmat_np[b][np.ix_(p_idx, t_idx)]
                _, sub_col_ind = linear_sum_assignment(sub_dist)

                # Map local solution back to global indices
                global_matched_indices = t_idx[sub_col_ind]
                batch_col_indices[p_idx] = global_matched_indices

            final_perm_indices.append(torch.from_numpy(batch_col_indices))

        perm_indices = torch.stack(final_perm_indices).to(device)

        # Reorder truth and calc MSE on [..., :3]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_tokens)
        truth_ordered = truth[batch_indices, perm_indices]

        return F.mse_loss(pred[..., :3], truth_ordered[..., :3], reduction='sum')


class TorchPermutationLossXY(torch.nn.Module):
    """
    GPU-accelerated loss that enforces split matching (Color 0<->0, 1<->1).
    Uses dynamic bucketing to handle variable agent counts efficiently on GPU
    without padding hacks or CPU loops.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, truth):
        device = pred.device
        total_loss = torch.tensor(0.0, device=device)

        for color in [0, 1]:
            mask_p = (pred[..., 3] == color)
            mask_t = (truth[..., 3] == color)

            counts_p = mask_p.sum(dim=1)
            unique_counts = torch.unique(counts_p)

            for k in unique_counts:
                k = k.item()
                if k == 0: continue

                # Identify batch items with exactly k agents
                batch_mask = (counts_p == k)

                # Extract and reshape to dense tensor [Subset_Batch, k, 4]
                sub_p = pred[batch_mask][mask_p[batch_mask]].view(-1, k, 4)
                sub_t = truth[batch_mask][mask_t[batch_mask]].view(-1, k, 4)

                # 1. Compute Cost Matrix (Euclidean squared on x,y)
                dist_sq = (sub_p[..., :2].unsqueeze(2) - sub_t[..., :2].unsqueeze(1)) ** 2
                dist_mat = dist_sq.sum(dim=-1)

                # FIX: Detach dist_mat for assignment solver (indices don't need grad)
                assignment = batch_linear_assignment(dist_mat.detach())
                _, col_ind = assignment_to_indices(assignment)
                col_ind = col_ind.to(device).long()

                # 3. Reorder Truth
                batch_idx = torch.arange(sub_p.shape[0], device=device).unsqueeze(1).expand_as(col_ind)
                sub_t_ordered = sub_t[batch_idx, col_ind]

                # 4. Compute MSE Loss (on x, y, angle)
                total_loss += F.mse_loss(sub_p[..., :3], sub_t_ordered[..., :3], reduction='sum')

        return total_loss


# ==========================================
# Use XY, Angle to calculate distance
# ==========================================
class ScipyPermutationXYA(torch.nn.Module):
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


class TorchPermutationXYA(torch.nn.Module):
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


# ==========================================
# 3. Profiling Harness
# ==========================================
def run_profile():
    MICRO = 10**6
    BATCH_SIZE = 32
    NOISE_MAGNITUDE = 1.
    N_VALUES = [3*2**8 for i in range(4, 9)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Initialize all 4 losses
    scipy_xyo = ScipyPermutationLossXY()
    scipy_xya = ScipyPermutationXYA()
    torch_xyo = TorchPermutationLossXY()
    torch_xya = TorchPermutationXYA()

    # Set seed to reproduce results
    torch.manual_seed(0)

    for N in N_VALUES:
        NONES = N // 3
        NZEROS = N - NONES
        print(f"Running N={N} (NZEROS={NZEROS}, NONES={NONES})")

        # TRUTH
        truth = torch.randn(BATCH_SIZE, N, 4, device=device, requires_grad=False)
        truth.data[..., 3] = torch.cat([torch.zeros(BATCH_SIZE, NZEROS), torch.ones(BATCH_SIZE, NONES)], dim=1).to(device)

        # PREDS
        preds_data = truth.clone()
        preds_data[..., :3] += torch.randn(BATCH_SIZE, N, 3, device=device) * NOISE_MAGNITUDE
        preds = preds_data.detach().requires_grad_(True)

        # 1. Vanilla MSE
        start = time.perf_counter()
        loss_base = F.mse_loss(preds[..., :3], truth[..., :3], reduction='sum')
        if device.type == 'cuda': torch.cuda.synchronize()
        time_base = (time.perf_counter() - start) * MICRO

        # 2. Scipy Original
        start = time.perf_counter()
        loss_sp_xyo = scipy_xyo(preds, truth)
        loss_sp_xyo.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        time_sp_xyo = (time.perf_counter() - start) * MICRO

        # 3. Scipy Simple
        start = time.perf_counter()
        loss_sp_xya = scipy_xya(preds, truth)
        loss_sp_xya.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        time_sp_xya = (time.perf_counter() - start) * MICRO

        # 4. Torch Original
        start = time.perf_counter()
        loss_tr_xyo = torch_xyo(preds, truth)
        loss_tr_xyo.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        time_tr_xyo = (time.perf_counter() - start) * MICRO

        # 5. Torch Simple
        start = time.perf_counter()
        loss_tr_xya = torch_xya(preds, truth)
        loss_tr_xya.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        time_tr_xya = (time.perf_counter() - start) * MICRO

        names = ["Basic", "Scipy XY-", "Torch XY-", "Scipy XYA", "Torch XYA"]

        compare(
            [loss_base, loss_sp_xyo, loss_tr_xyo, loss_sp_xya, loss_tr_xya],
            names, "Loss", "up"
        )
        compare(
            [time_base, time_sp_xyo, time_tr_xyo, time_sp_xya, time_tr_xya],
            names, "Time (μs)", "down"
        )

if __name__ == "__main__":
    run_profile()

"""
Running on: cuda
Running N=768 (NZEROS=512, NONES=256)
┌───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│Loss       │Basic      │Scipy XY-  │Torch XY-  │Scipy XYA  │Torch XYA  │
├───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│Basic      │    (73959)│       0.89│       0.89│        3.5│        3.5│  # XY Only > Basic > XYA
│Scipy XY-  │           │    (82882)│       1.00│        4.0│        4.0│
│Torch XY-  │           │           │    (82882)│        4.0│        4.0│
│Scipy XYA  │           │           │           │    (20949)│       1.00│  # Torch === Scipy
│Torch XYA  │           │           │           │           │    (20949)│
└───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
┌───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│Time (μs)  │Basic      │Scipy XY-  │Torch XY-  │Scipy XYA  │Torch XYA  │
├───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│Basic      │      (161)│           │           │           │           │
│Scipy XY-  │       8162│  (1316984)│           │           │           │
│Torch XY-  │      33895│        4.2│  (5469341)│           │           │ # Torch version is 4 times slower on GPU
│Scipy XYA  │       6258│       0.77│       0.18│  (1009773)│           │
│Torch XYA  │      25446│        3.1│       0.75│        4.1│  (4105996)│
└───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘

Running on: cpu
Running N=768 (NZEROS=512, NONES=256)
┌───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│Loss       │Basic      │Scipy XY-  │Torch XY-  │Scipy XYA  │Torch XYA  │
├───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│Basic      │    (73855)│       0.88│       0.88│        3.5│        3.5│
│Scipy XY-  │           │    (83768)│       1.00│        4.0│        4.0│
│Torch XY-  │           │           │    (83768)│        4.0│        4.0│
│Scipy XYA  │           │           │           │    (20993)│       1.00│
│Torch XYA  │           │           │           │           │    (20993)│
└───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
┌───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│Time (μs)  │Basic      │Scipy XY-  │Torch XY-  │Scipy XYA  │Torch XYA  │
├───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│Basic      │      (156)│           │           │           │           │
│Scipy XY-  │       9161│  (1429721)│           │           │           │
│Torch XY-  │       8026│       0.88│  (1252589)│           │           │ # Torch version is 20% faster on GPU
│Scipy XYA  │      10036│       1.10│       1.25│  (1566279)│           │
│Torch XYA  │       8085│       0.88│       1.01│       0.81│  (1261850)│
└───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
"""