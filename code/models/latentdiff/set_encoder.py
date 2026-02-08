import torch
import torch.nn as nn

from .mhab import MultiheadAttentionBlock

class SetEncoder(nn.Module):
    def __init__(self, num_classes, latent_dim, num_pools, num_blocks, num_heads):
        super().__init__()
        dim = latent_dim // num_pools
        self.num_pools = num_pools
        assert latent_dim == num_pools * dim, "latent_dim must be divisible by num_pools"

        self.pointwise_mlp = nn.Sequential(
            nn.Linear(4, dim),              # xyac
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.self_attn_blocks = nn.ModuleList([
            MultiheadAttentionBlock(dim, num_heads)
            for _ in range(num_blocks)
        ])

        self.class_embed = nn.Embedding(num_classes + 1, latent_dim)
        self.attn_pool = MultiheadAttentionBlock(dim, num_heads)

        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, color, cls):
        B, N, THREE = x.shape
        color = color.unsqueeze(-1)              # B, N, 1
        h = torch.cat([x, color], dim=-1)        # B, N, 4
        h = self.pointwise_mlp(h)                # B, N, D
        for sab in self.self_attn_blocks:
            h = sab(h, h)

        # Class Embedding acts as inital Seeds
        cemb = self.class_embed(cls)                # B, P*D
        seeds = cemb.reshape(B, self.num_pools, -1) # B, P, D

        # We do num_pools and concatenate them
        h = self.attn_pool(seeds, h)                # B, P, D
        h = h.reshape(B, -1)                        # B, P*D

        mu = self.mu_head(h)                     # B, L
        logvar = self.logvar_head(h)             # B, L
        return mu, logvar

