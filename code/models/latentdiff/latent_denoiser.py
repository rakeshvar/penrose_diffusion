import torch.nn as nn

from code.models.sinusoidal import SinusoidalPositionalEmbedding

#------------------------------------------------------------
# Vanilla MLP Denoiser
#------------------------------------------------------------
class MLPLatentDenoiser(nn.Module):
    def __init__(self, num_classes, latent_dim, num_blocks, T=1000):
        super().__init__()
        self.D = latent_dim

        self.time_embed = nn.Embedding(T, latent_dim)
        self.class_embed = nn.Embedding(num_classes + 1, latent_dim) # CFG

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, z_t, t, cls):
        h = z_t + self.time_embed(t) + self.class_embed(cls)
        return self.net(h)

    @property
    def device(self):
        return next(self.parameters()).device

#------------------------------------------------------------
# FiLM Denoiser
#------------------------------------------------------------
class FiLMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.linear = nn.Linear(in_dim, out_dim)

        # Learn scale and shift from conditioning
        self.film = nn.Linear(cond_dim, out_dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.act = nn.SiLU()

        # Residual connection
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, cond):
        # Normalize
        h = self.norm(x)

        # Apply linear
        h = self.linear(h)

        # FiLM modulation
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + scale) + shift

        # Activation and residual
        h = self.act(h)
        return h + self.residual(x)


class FiLMLatentDenoiser(nn.Module):
    def __init__(self, num_classes, latent_dim, num_blocks):
        super().__init__()
        self.dim = time_emb_dim = hidden_dim = latent_dim

        # Sinusoidal time embedding (critical!)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.class_embed = nn.Embedding(num_classes + 1, hidden_dim)

        # Deeper network with FiLM conditioning
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_blocks)
        ])
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, t, cls):
        t_emb = self.time_mlp(t)
        c_emb = self.class_embed(cls)
        cond = t_emb + c_emb

        h = self.input_proj(z_t)
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h)

    @property
    def device(self):
        return next(self.parameters()).device