import torch
from torch import nn
from ..sinusoidal import SinusoidalPositionalEmbedding

class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        num_classes: int,     # 70
        class_embed_dim: int,
        time_embed_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        io_dim: int=4,
        predict: str='noise',
        num_global_tokens: int=4,
        **ignore
    ):
        super().__init__()
        self.predict = predict
        self.d_model = d_model
        self.io_dim = io_dim
        self.num_global_tokens = num_global_tokens

        # Learnable global tokens
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, d_model))

        # Input projection
        self.input_proj = nn.Linear(io_dim, d_model)

        # Color embedding (Binary: 0 or 1)
        self.color_embed = nn.Embedding(2, d_model)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, d_model)
        )

        # Class embedding (one-hot + linear projection)
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.class_proj = nn.Linear(class_embed_dim, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Layer norm for output
        self.norm_out = nn.LayerNorm(d_model)
        # Output projection - predict noise for x, y, sin, cos
        # Only applied to tile tokens, not global tokens
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, io_dim)
        )


    def forward(self, xysc, colors, t, class_labels):
        """
        Args:
            xysc: (B, N, 4) - noisy tiles [x, y, sin, cos]
            colors: (B, N)
            t: (B,) - time steps unnormalized
            class_labels: (B,) - class indices (0-69)
        Returns:
            noise: (B, N, 4) - predicted noise for [x, y, sin, cos]
        """
        B, _, _ = xysc.shape

        # project to d_model
        h = self.input_proj(xysc)                           # B, N, D

        # color embedding
        h_color = self.color_embed(colors)                  # B, N, D
        h = h + h_color

        # global tokens
        global_tokens = self.global_tokens.expand(B, -1, -1)  # B, G, D

        # Concatenate: [global_tokens, tile_tokens]
        h = torch.cat([global_tokens, h], dim=1)            # B, G+N, D

        # time & class
        time_emb = self.time_embed(t).unsqueeze(1)          # B, 1, D
        class_emb = self.class_embed(class_labels)          # B, class_embed_dim
        class_emb = self.class_proj(class_emb).unsqueeze(1) # B, 1, D
        h = h + time_emb + class_emb

        # process
        h = self.transformer(h)                             # B, G+N, D

        # Extract only the tile tokens (skip global tokens)
        h_tiles = h[:, self.num_global_tokens:, :]          # B, N, D

        # Normalize and project to noise
        h_tiles = self.norm_out(h_tiles)
        noise_pred = self.output_proj(h_tiles)              # B, N, 4

        return noise_pred

    @property
    def device(self):
        return next(self.parameters()).device