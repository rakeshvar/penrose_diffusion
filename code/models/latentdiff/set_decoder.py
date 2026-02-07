import torch
import torch.nn as nn

# ------------------------------------------------------------
# Perceiver-style Decoder
# ------------------------------------------------------------

class PerceiverBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, z):
        z = z.unsqueeze(1)                      # (B, D) -> (B, 1, D)
        h, _ = self.cross_attn(x, z, z) # x <- z
        x = self.norm1(x + h)

        h, _ = self.self_attn(x, x, x)  # x <- x
        x = self.norm2(x + h)

        h = self.ff(x)
        x = self.norm3(x + h)
        
        return x


class PerceiverDecoder(nn.Module):
    def __init__(self, num_tiles, latent_dim, num_blocks, num_heads):
        super().__init__()

        self.seeds = nn.Parameter(torch.randn(1, num_tiles, latent_dim)) # 1, N, D

        self.color_embedding = nn.Embedding(2, latent_dim)
        self.blocks = nn.ModuleList([
            PerceiverBlock(latent_dim, num_heads) for _ in range(num_blocks)
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
        xya = self.seeds.expand(B, -1, -1)              # 1, N, D → B, N, D
        c_emb = self.color_embedding(color.long())      # B, N    → B, N, D
        xya = xya + c_emb

        for blk in self.blocks:
            xya = blk(xya, z)

        xya = self.output_mlp(xya)

        return xya