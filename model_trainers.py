from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import compatibility as compat

from model_ddim import DDIMDiffuser, TransformerDenoiser
from model_losses import PermutationLossTorch


class AbstractTrainer(ABC):
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


class NoisePredictor(AbstractTrainer):
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        # L2 Loss on noise
        return F.mse_loss(noise, noise_hat)


class SamplePredictor(AbstractTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser.predict = 'sample'  # Important

    def compute_loss(self, xysc_0, xysc_t, noise, xysc_0_hat, colors, t):
        # L2 Loss on sample (prediction is sample)
        return F.mse_loss(xysc_0, xysc_0_hat)


class LSASerial(AbstractTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lossfn = PermutationLossTorch()

    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        # Recover sample from noise prediction
        xysc_recovered = self.diffuser.recover_xysc(xysc_t, t, noise_hat)
        
        # LSA Loss
        return self.lossfn(xysc_0, xysc_recovered, colors)

#------------------------------------------------------------------------------
# LSAParallel - Do from scratch
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


class LSAParallel(AbstractTrainer):
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

        # Loss
        # -  Ensure denoiser is done 
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream_denoiser)

        loss = F.mse_loss(noise[bi, ti], noise_hat[bi, pi])

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        compat.optimizer_step(self.optimizer)

        return loss.item()
