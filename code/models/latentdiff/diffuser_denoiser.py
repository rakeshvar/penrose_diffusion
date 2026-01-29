
import torch
import torch.nn as nn
from code.models.diffuser import Diffuser


class LatentDiffuser(Diffuser):
    @torch.no_grad()
    def sample(self, denoiser, labels, num_steps=50, guidance_scale=2.0):
        B = labels.shape[0]
        D = denoiser.time_embed.embedding_dim
        device = denoiser.device
        NULL = denoiser.class_embed.num_embeddings - 1
        nulls = torch.full_like(labels, NULL)

        z = torch.randn((B, D), device=device)
        times = torch.linspace(self.num_timesteps-1, 0, num_steps+1, device=device).long()

        for i in range(num_steps):
            t = torch.full((B,), times[i], device=device, dtype=torch.long) # type: ignore
            ε_cond = denoiser(z, t, labels)
            ε_null = denoiser(z, t, nulls)
            ε_hatt = (1 + guidance_scale) * ε_cond - guidance_scale * ε_null
            z = self.p_sample(z, ε_hatt, t)

        return z

# ------------------------------------------------------------
# Latent Denoiser (CFG-capable)
# ------------------------------------------------------------

class LatentDenoiser(nn.Module):
    def __init__(self, D, num_classes, T=1000):
        super().__init__()
        self.time_embed = nn.Embedding(T, D)
        self.class_embed = nn.Embedding(num_classes + 1, D) # CFG

        self.net = nn.Sequential(
            nn.Linear(D, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, D),
        )

    def forward(self, z_t, t, cls):
        h = z_t + self.time_embed(t) + self.class_embed(cls)
        return self.net(h)
    
    @property
    def device(self):
        return next(self.parameters()).device

