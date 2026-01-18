from abc import ABC

import torch
import torch.nn.functional as F
import code.compatibility as compat

from code.model.ddim import DDIMDiffuser, TransformerDenoiser

#------------------------------------------------------------------------------
# Registry of losses
#------------------------------------------------------------------------------
_LOSS_REGISTRY = {}

def register_loss(*aliases):
    """    Decorator to register a loss class with multiple name aliases. """
    def decorator(cls):
        cls._canonical_name = aliases[0].lower() if aliases else cls.__name__.lower()
        
        for alias in aliases:
            _LOSS_REGISTRY[alias.lower()] = cls
        return cls
    return decorator


#------------------------------------------------------------------------------
# Abstract Base Class for Losses
#------------------------------------------------------------------------------
class AbstractLoss(ABC):
    def __init__(self,
                 denoiser: TransformerDenoiser,
                 diffuser: DDIMDiffuser,
                 optimizer,
                 device):
        self.denoiser = denoiser
        self.optimizer = optimizer
        self.diffuser = diffuser
        self.device = device

    def __call__(self, xysc_0, colors, labels):
        self.denoiser.train()
        B = xysc_0.shape[0]

        # Forward pass - Add Noise
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        xysc_t, noise = self.diffuser.q_sample(xysc_0, t)

        # Predict noise/sample
        prediction = self.denoiser(xysc_t, colors, t.float(), labels)

        # Compute loss (subclass-specific)
        loss = self.compute_loss(xysc_0, xysc_t, noise, prediction, colors, t) # type: ignore

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()

        # Universal Step (TPU/GPU)
        compat.optimizer_step(self.optimizer)

        return loss.item()

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device})"
    
    @property
    def canonical_name(self):
        """Returns the canonical short name for this loss."""
        return getattr(self.__class__, '_canonical_name', self.__class__.__name__.lower())
    
    
#------------------------------------------------------------------------------
# NoisePredictionLoss
#------------------------------------------------------------------------------
@register_loss('npl', 'noise', 'n')
class NoisePredictionLoss(AbstractLoss):
    def compute_loss(self, xysc_0, xysc_t, noise, prediction, colors, t):
        # L2 Loss on noise prediction
        noise_hat = prediction
        return F.mse_loss(noise, noise_hat)

#------------------------------------------------------------------------------
# SamplePredictionLoss
#------------------------------------------------------------------------------
@register_loss('spl', 'sample', 'sp')
class SamplePredictionLoss(AbstractLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser.predict = 'sample'  # Important

    def compute_loss(self, xysc_0, xysc_t, noise, prediction, colors, t):
        # L2 Loss on sample prediction
        xysc_0_hat = prediction
        return F.mse_loss(xysc_0, xysc_0_hat)

#------------------------------------------------------------------------------
# SamplePredictionLoss
#------------------------------------------------------------------------------
@register_loss('sal', 'sa', 'sampleangle', 'sangle', 'sampang')
class SampleAngleLoss(AbstractLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser.predict = 'sample'  # Important

    def compute_loss(self, xysc_0, xysc_t, noise, prediction, colors, t):
        xysc_0_hat = prediction
        sample_loss = F.mse_loss(xysc_0, xysc_0_hat)

        sc = xysc_0_hat[..., -2:]                                   # B, N, 2
        circle_loss = ((sc.pow(2).sum(dim=-1) - 1.0) ** 2).mean()   # B, N
        mean_sc = sc.mean(dim=1, keepdim=True)                      # B, 1, 2
        equal_angle_loss = ((sc - mean_sc) ** 2).mean()

        return sample_loss + circle_loss + equal_angle_loss

#------------------------------------------------------------------------------
# Linear Sum Assignment Losss (Torch)
#------------------------------------------------------------------------------
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


@register_loss('lsas', 'lsaserial', 'serial')
class LSALossSerial(AbstractLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        # Recover sample from predicted noise
        xysc_0_hat = self.diffuser.recover_xysc(xysc_t, t, noise_hat)

        if not compat.IS_TPU:
            return lsa_loss_cuda(xysc_0, xysc_0_hat, colors)

        else:
            # Cost Matrix for LSA
            diff = xysc_0_hat.unsqueeze(2) - xysc_0.unsqueeze(1)
            cost_matrix = (diff ** 2).sum(dim=-1)
            cost_np = cost_matrix.detach().cpu().numpy()
            colors_np = colors.detach().cpu().numpy()

            bi, ti, pi = lsa_ordering_np(cost_np, colors_np)
            with torch.no_grad():
                bi = torch.from_numpy(bi).to(self.device, non_blocking=True)
                ti = torch.from_numpy(ti).to(self.device, non_blocking=True)
                pi = torch.from_numpy(pi).to(self.device, non_blocking=True)

            return F.mse_loss(xysc_0[bi, ti], xysc_0_hat[bi, pi])

#------------------------------------------------------------------------------
# Linear Sum Assignment (Parallel)
#------------------------------------------------------------------------------
from contextlib import nullcontext
import numpy as np
from scipy.optimize import linear_sum_assignment


def lsa_ordering_np(cost_np, colors_np):
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


def maybe_stream(stream, enabled):
    return torch.cuda.stream(stream) if enabled else nullcontext()


@register_loss('lsap', 'lsaparallel', 'parallel')
class LSALossParallel(AbstractLoss):
    """
    Loss calculation:
        xysc_0 is permuted so that it is closest to xysc_t
        MSE ( xysc_0_permuted, xysc_0_hat )
        Equivalently
        MSE (noise_permuted, noise_hat)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_cuda = (self.device.type == "cuda")
        self.stream_denoiser = torch.cuda.Stream() if self.use_cuda else None
        self.stream_cost = torch.cuda.Stream() if self.use_cuda else None

    def __call__(self, xysc_0, colors, labels):
        self.denoiser.train()
        B = xysc_0.shape[0]

        # Forward pass - Add Noise
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        xysc_t, noise = self.diffuser.q_sample(xysc_0, t)

        # Predict noise
        with maybe_stream(self.stream_denoiser, self.use_cuda):
            noise_hat = self.denoiser(xysc_t, colors, t.float(), labels)
            xysc_0_hat = self.diffuser.recover_xysc(xysc_t, t, noise_hat)

        # Cost Matrix for LSA
        with maybe_stream(self.stream_cost, self.use_cuda):
            diff = xysc_t.unsqueeze(2) - xysc_0.unsqueeze(1)
            cost_matrix = (diff ** 2).sum(dim=-1)

        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream_cost)

        cost_np = cost_matrix.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy()

        bi, ti, pi = lsa_ordering_np(cost_np, colors_np)

        with torch.no_grad():
            bi = torch.from_numpy(bi).to(self.device, non_blocking=True)
            ti = torch.from_numpy(ti).to(self.device, non_blocking=True)
            pi = torch.from_numpy(pi).to(self.device, non_blocking=True)

        # Ensure denoiser is done
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream_denoiser)

        # Loss
        loss = F.mse_loss(xysc_0[bi, ti], xysc_0_hat[bi, pi])

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        compat.optimizer_step(self.optimizer)

        return loss.item()
    

#------------------------------------------------------------------------------
# Factory
#------------------------------------------------------------------------------
def get_loss(name: str, *args, **kwargs):
    """     Factory function to instantiate a loss by name (case-insensitive).    """
    key = name.lower()
    if key not in _LOSS_REGISTRY:
        # Build helpful error message
        unique_classes = sorted(set(_LOSS_REGISTRY.values()), key=lambda x: x.__name__)
        available = {cls.__name__: [k for k, v in _LOSS_REGISTRY.items() if v == cls] 
                     for cls in unique_classes}
        
        error_msg = f"Loss '{name}' not found. Available losses:\n"
        for cls_name, alias_list in available.items():
            error_msg += f"  {cls_name}: {', '.join(alias_list)}\n"
        
        raise ValueError(error_msg)
    
    loss_class = _LOSS_REGISTRY[key]
    return loss_class(*args, **kwargs)


def list_losses():
    """Returns a dict mapping loss class names to their aliases."""
    unique_classes = sorted(set(_LOSS_REGISTRY.values()), key=lambda x: x.__name__)
    return {cls.__name__: sorted([k for k, v in _LOSS_REGISTRY.items() if v == cls])
            for cls in unique_classes}
