import torch
import torch.nn as nn

# ------------------------------------------------------------
# Attention pooling (encoder)
# ------------------------------------------------------------
class MultiHeadAttnPool(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.pool_token = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, h):
        B, N, D = h.shape
        q = self.pool_token.expand(B, 1, -1)     # B, 1, D
        out, _ = self.attn(q, h, h)              # B, 1, D
        return out.squeeze(1)                    # B, D

# ------------------------------------------------------------
# Encoder (Set → latent)
# ------------------------------------------------------------
class SetEncoder(nn.Module):
    def __init__(self, D, num_classes, heads=4):
        super().__init__()

        self.point_mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, D),
            nn.ReLU(),
        )

        self.pool = MultiHeadAttnPool(D, heads=heads)
        self.class_embed = nn.Embedding(num_classes + 1, D)
        self.fc_mu = nn.Linear(D, D)
        self.fc_logvar = nn.Linear(D, D)

    def forward(self, x, color, cls):
        color = color.unsqueeze(-1)              # B, N, 1
        h = torch.cat([x, color], dim=-1)        # B, N, 4
        h = self.point_mlp(h)                    # B, N, D

        h = self.pool(h)                         # B, D
        h = h + self.class_embed(cls)            # B, D

        mu = self.fc_mu(h)                       # B, D
        logvar = self.fc_logvar(h)               # B, D

        return mu, logvar

