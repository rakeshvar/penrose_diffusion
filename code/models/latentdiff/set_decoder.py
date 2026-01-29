import torch
import torch.nn as nn

# ------------------------------------------------------------
# Perceiver-style Decoder
# ------------------------------------------------------------

class PerceiverBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, q, z):
        z = z.unsqueeze(1)
        h, _ = self.cross_attn(q, z, z)
        q = self.norm1(q + h)

        h, _ = self.self_attn(q, q, q)
        q = self.norm2(q + h)

        q = self.norm3(q + self.ff(q))
        return q


class PerceiverDecoder(nn.Module):
    def __init__(self, latent_dim, num_tiles):
        super().__init__()

        self.queries = nn.Parameter(torch.randn(num_tiles, latent_dim))

        self.blocks = nn.ModuleList([
            PerceiverBlock(latent_dim),
            PerceiverBlock(latent_dim),
        ])

        self.output_mlp = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, z, color):
        B, N, _ = color.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)

        for blk in self.blocks:
            q = blk(q, z)

        return self.output_mlp(torch.cat([q, color], dim=-1))

