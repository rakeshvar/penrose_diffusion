import torch
import torch.nn as nn
from ..sinusoidal import SinusoidalPositionalEmbedding

class MAB(nn.Module):
    """
    Multihead Attention Block (The 'Transformer Block' of Set Transformers)
    Performs: X = LayerNorm(X + Attn(X, Y)) -> X = LayerNorm(X + FF(X))
    """
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, Q, K, V):
        # Attention + Residual + Norm
        # Note: PyTorch MHA returns (attn_output, attn_weights)
        attn_out, _ = self.mha(Q, K, V)
        X = self.norm1(Q + attn_out)
        
        # Feedforward + Residual + Norm
        return self.norm2(X + self.ff(X))

class ISABlock(nn.Module):
    """
    Induced Set Attention Block
    Reduces complexity from O(N^2) to O(NM) using M inducing points.
    """
    def __init__(self, dim, num_inducing, num_heads, dropout=0.0):
        super().__init__()
        # Learnable Inducing Points
        self.I = nn.Parameter(torch.Tensor(1, num_inducing, dim))
        nn.init.xavier_uniform_(self.I)
        
        # Two MAB steps:
        # I = X -> I
        self.mab0 = MAB(dim, num_heads, dropout)
        # X = I -> X
        self.mab1 = MAB(dim, num_heads, dropout)

    def forward(self, X):
        B = X.shape[0]
        I = self.I.expand(B, -1, -1)    # (1, M, D) -> (B, M, D)
        H = self.mab0(I, X, X)          # (B, M, D) - The "Bottleneck" 
        X = self.mab1(X, H, H)       # (B, N, D) - The "Broadcast"
        return X

class ISABDenoiser(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_embed_dim: int,
        time_embed_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        io_dim: int=4,
        predict: str='noise',
        num_inducing: int=16,
        **ignore
    ):
        super().__init__()
        self.predict = predict
        self.d_model = d_model
        self.io_dim = io_dim
        self.num_inducing = num_inducing

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

        # The ISAB Stack
        # Replaces nn.TransformerEncoder
        self.layers = nn.ModuleList([
            ISABlock(d_model, num_inducing, num_heads, dropout) 
            for _ in range(num_layers)
        ])

        # Output Projection
        self.norm_out = nn.LayerNorm(d_model)
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
        xysc: (B, N, 4)
        colors: (B, N)
        t: (B,)
        class_labels: (B,)
        """
        # project to d_model        
        h = self.input_proj(xysc)                           # B, N, D

        # color embedding
        h_color = self.color_embed(colors)                  # B, N, D
        h = h + h_color

        # time & class
        # We broadcast (B, 1, D) to (B, N, D) so every point knows the context
        time_emb = self.time_embed(t).unsqueeze(1)
        class_emb = self.class_embed(class_labels)
        class_emb = self.class_proj(class_emb).unsqueeze(1)
        h = h + time_emb + class_emb

        # Process through ISAB layers (Permutation Invariant)
        for layer in self.layers:
            h = layer(h) # Dimensions stay (B, N, D)

        # Final Prediction
        h = self.norm_out(h)
        noise_pred = self.output_proj(h)
        
        return noise_pred

    @property
    def device(self):
        return next(self.parameters()).device