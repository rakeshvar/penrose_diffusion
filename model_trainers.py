
from abc import ABC

import torch
import torch.nn.functional as F

from model_ddim import DDIMDiffusion, TransformerDenoiser
from model_losses import PermutationLossTorch, PermutationOnlyScipy

class AbstractTrainer(ABC):
    def __init__(self,
                 denoiser: TransformerDenoiser,
                 diffuser: DDIMDiffusion,
                 optimizer,
                 device):
        self.denoiser = denoiser
        self.optimizer = optimizer
        self.diffuser = diffuser
        self.device = device


class NoisePredictor(AbstractTrainer):
    def __call__(self, xysc, colors, labels):
        self.denoiser.train()
        B = xysc.shape[0]

        # Forward pass — Add Noise
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        xysc_noisy, noise = self.diffuser.q_sample(xysc, t)

        # Predict noise
        noise_pred = self.denoiser(xysc_noisy, colors, t.float(), labels)
        loss = F.mse_loss(noise, noise_pred)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class XYAPredictor(AbstractTrainer):
    def __call__(self, xysc, colors, labels):
        self.denoiser.train()
        B = xysc.shape[0]

        # Forward pass
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        xysc_noisy, noise = self.diffuser.q_sample(xysc, t)

        # Predict noise
        xysc_pred = self.denoiser(xysc_noisy, colors, t.float(), labels)
        loss = F.mse_loss(xysc, xysc_pred)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class LSASerial(AbstractTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lossfn = PermutationLossTorch()

    def __call__(self, xysc, colors, labels):
        self.denoiser.train()
        B = xysc.shape[0]

        # Forward pass
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        xysc_noisy, noise = self.diffuser.q_sample(xysc, t)

        # Predict noise
        noise_pred = self.denoiser(xysc_noisy, colors, t.float(), labels)
        xysc_recovered = xysc_noisy - noise_pred

        # LSA Loss
        loss = self.lossfn(xysc, xysc_recovered, colors)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class LSAParallel(AbstractTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orderfn = PermutationOnlyScipy()

    def __call__(self, xysc, colors, labels):
        self.denoiser.train()
        B = xysc.shape[0]

        # Forward pass
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        xysc_noisy, noise = self.diffuser.q_sample(xysc, t)

        # Predict noise
        noise_pred = self.denoiser(xysc_noisy, colors, t.float(), labels)
        xysc_recovered = xysc_noisy - noise_pred

        # Reorder
        bi, pi, ti = self.orderfn(xysc, xysc_noisy, colors)

        # LSA Loss
        loss = F.mse_loss(xysc[bi, ti], xysc_recovered[bi, pi])

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

