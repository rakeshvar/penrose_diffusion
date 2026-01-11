
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from model_losses import PermutationLossTorch, PermutationLossScipy

class AbstractTrainer(ABC):
    def __init__(self, denoiser, optimizer, diffuser, device):
        self.denoiser = denoiser
        self.optimizer = optimizer
        self.diffuser = diffuser
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class NoisePredictor(AbstractTrainer):
    def __call__(self, xyac, labels):
        self.denoiser.train()
        B = xyac.shape[0]

        # Forward pass — Add Noise
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(xyac[..., :3])
        xyac_noisy, noise_target = self.diffuser.q_sample(xyac, t, noise)

        # Predict noise
        noise_pred = self.denoiser(xyac_noisy, t.float() / self.diffuser.num_timesteps, labels)
        loss = F.mse_loss(noise_pred, noise_target)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class XYACPredictor(AbstractTrainer):
    def __call__(self, xyac, labels):
        self.denoiser.train()
        B = xyac.shape[0]

        # Forward pass
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(xyac[..., :3])
        xyac_noisy, _noise_target = self.diffuser.q_sample(xyac, t, noise)

        # Predict noise
        xyac_pred = self.denoiser(xyac_noisy, t.float() / self.diffuser.num_timesteps, labels)
        loss = F.mse_loss(xyac_pred, xyac_noisy)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class LSASerial(AbstractTrainer):
    def __call__(self, xyac, labels):
        self.denoiser.train()
        B = xyac.shape[0]

        # Forward pass
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(xyac[..., :3])
        xyac_noisy, noise_target = self.diffuser.q_sample(xyac, t, noise)

        # Predict noise
        noise_pred = self.denoiser(xyac_noisy, t.float() / self.diffuser.num_timesteps, labels)
        xyac_recovered = xyac_noisy - noise_pred

        # LSA Loss
        loss = PermutationLossTorch()(xyac_recovered, xyac_noisy)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class LSAParallel(AbstractTrainer):
    def __call__(self, xyac, labels):
        self.denoiser.train()
        B = xyac.shape[0]

        # Forward pass
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(xyac[..., :3])
        xyac_noisy, noise_target = self.diffuser.q_sample(xyac, t, noise)

        # Predict noise
        noise_pred = self.denoiser(xyac_noisy, t.float() / self.diffuser.num_timesteps, labels)
        xyac_recovered = xyac_noisy - noise_pred

        # LSA Loss
        loss = PermutationLossTorch()(xyac_recovered, xyac_noisy)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()