from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import compatibility as compat

from model_ddim import DDIMDiffuser, TransformerDenoiser
from model_losses import PermutationLossTorch, PermutationOnlyScipy


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
    def compute_loss(self, xysc0, xysc_t, noise, noise_pred, colors, t):
        # L2 Loss on noise
        return F.mse_loss(noise, noise_pred)


class SamplePredictor(AbstractTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser.predict = 'sample'  # Important

    def compute_loss(self, xysc, xysc_t, noise, xysc_pred, colors, t):
        # L2 Loss on sample (prediction is sample)
        return F.mse_loss(xysc, xysc_pred)


class LSASerial(AbstractTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lossfn = PermutationLossTorch()

    def compute_loss(self, xysc, xysc_t, noise, noise_pred, colors, t):
        # Recover sample from noise prediction
        xysc_recovered = self.diffuser.recover_xysc(xysc_t, t, noise_pred)
        
        # LSA Loss
        return self.lossfn(xysc, xysc_recovered, colors)


class LSAParallel(AbstractTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orderfn = PermutationOnlyScipy()

    def compute_loss(self, xysc, xysc_t, noise, noise_pred, colors, t):
        # Recover sample from noise prediction
        xysc_recovered = self.diffuser.recover_xysc(xysc_t, t, noise_pred)

        # Reorder using scipy-based assignment
        bi, ti, pi = self.orderfn(xysc, xysc_t, colors)

        # L2 Loss on reordered xysc
        return F.mse_loss(xysc[bi, ti], xysc_recovered[bi, pi])