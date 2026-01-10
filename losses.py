import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices


def pairwise_compare(values, names, metric_name, direction="down"):
    """
    Simple utility to print comparison table.
    direction: "down" means lower is better (Time), "up" means higher is better (Score/Loss)
    """
    print(f"\n--- {metric_name} Comparison ---")
    best_val = min(values) if direction == "down" else max(values)
    
    # Header
    print(f"{'Method':<15} | {'Value':<12} | {'vs Best':<10}")
    print("-" * 43)
    
    for val, name in zip(values, names):
        if val == best_val:
            diff_str = "(Best)"
        else:
            if direction == "down":
                diff = (val - best_val) / best_val * 100
                diff_str = f"+{diff:.1f}%"
            else:
                diff = (best_val - val) / best_val * 100
                diff_str = f"-{diff:.1f}%"
        
        # Determine format based on magnitude
        if abs(val) < 0.01:
            val_str = f"{val:.2e}"
        else:
            val_str = f"{val:.4f}"
            
        print(f"{name:<15} | {val_str:<12} | {diff_str:<10}")
    print("-" * 43)
    
# ==========================================
# 1. Original Classes (Reference)
# ==========================================
class ScipyPermutationLoss(torch.nn.Module):
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


class TorchPermutationLoss(torch.nn.Module):
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
# 2. New Simplified Classes
# ==========================================
class ScipyPermutationSimple(torch.nn.Module):
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


class TorchPermutationSimple(torch.nn.Module):
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
    N_VALUES = [2**i for i in range(5, 11)] 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # Initialize all 4 losses
    scipy_orig = ScipyPermutationLoss()
    scipy_simple = ScipyPermutationSimple()
    torch_orig = TorchPermutationLoss()
    torch_simple = TorchPermutationSimple()

    names = ["Basic", "SpOrig", "TrOrig", "SpSimp", "TrSimp"]

    for N in N_VALUES:
        NONES = N // 3
        NZEROS = N - NONES
        print(f"Running N={N} (NZEROS={NZEROS}, NONES={NONES})")

        # TRUTH is fixed (requires_grad=False)
        truth = torch.randn(BATCH_SIZE, N, 4, device=device, requires_grad=False)
        truth.data[..., 3] = torch.cat([torch.zeros(BATCH_SIZE, NZEROS), torch.ones(BATCH_SIZE, NONES)], dim=1).to(device)
        
        # PREDS is trainable (requires_grad=True)
        # We clone truth, then make it a leaf variable that requires grad
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
        loss_so = scipy_orig(preds, truth)
        loss_so.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        time_so = (time.perf_counter() - start) * MICRO

        # 3. Scipy Simple
        start = time.perf_counter()
        loss_ss = scipy_simple(preds, truth)
        loss_ss.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        time_ss = (time.perf_counter() - start) * MICRO

        # 4. Torch Original
        start = time.perf_counter()
        loss_to = torch_orig(preds, truth)
        loss_to.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        time_to = (time.perf_counter() - start) * MICRO

        # 5. Torch Simple
        start = time.perf_counter()
        loss_ts = torch_simple(preds, truth)
        loss_ts.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        time_ts = (time.perf_counter() - start) * MICRO

        pairwise_compare(
            [loss_base, loss_so, loss_to, loss_ss, loss_ts], 
            names, "Loss", "up"
        )
        pairwise_compare(
            [time_base, time_so, time_to, time_ss, time_ts], 
            names, "Time (μs)", "down"
        )

if __name__ == "__main__":
    run_profile()