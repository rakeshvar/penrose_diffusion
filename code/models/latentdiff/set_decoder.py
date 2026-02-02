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
        z = z.unsqueeze(1)                      # (B, D) -> (B, 1, D)
        h, _ = self.cross_attn(q, z, z) # q <- z
        q = self.norm1(q + h)

        h, _ = self.self_attn(q, q, q)  # q <- q
        q = self.norm2(q + h)

        q = self.norm3(q + self.ff(q))
        return q


class PerceiverDecoder(nn.Module):
    def __init__(self, latent_dim, num_tiles, num_blocks=2):
        super().__init__()

        self.queries = nn.Parameter(torch.randn(num_tiles, latent_dim)) # N, D

        self.color_embedding = nn.Embedding(2, latent_dim)
        self.blocks = nn.ModuleList([
            PerceiverBlock(latent_dim) for _ in range(num_blocks)
        ])

        self.output_mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),             # 1 for color
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, z, color):
        B, N = color.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)     # B, N, D
        c_emb = self.color_embedding(color.long())

        q = q + c_emb
        for blk in self.blocks:
            q = blk(q, z)


        # 5. Output
        xyac = self.output_mlp(q)

        return xyac