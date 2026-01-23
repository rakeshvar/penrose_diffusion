# pyright: reportIndexIssue=false
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from code.compatibility import maybe_mark_step

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for time"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) integer diffusion timesteps
        returns: (B, dim) sinusoidal time embedding
        """
        device = t.device
        n = self.dim // 2
        i = torch.arange(n, device=device)          # 1000 = num_timesteps
        ω = (1000 * math.pi) ** (-i/(n-1))          # Frequency ladder: ωᵢ = 1 / (πT)ⁱ⁾ⁿ
        Φ = t[:, None] * ω[None, :]                 # Phase: Φ[m, i] = ω[i] · t[m] = C^(-i/H) tₘ
        sinΦ, cosΦ = torch.sin(Φ), torch.cos(Φ)     # sin(ωᵢt) = sin(t/(πT)ⁱ⁾ⁿ)
        return torch.cat([sinΦ, cosΦ], dim=-1)

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
        predict: str='noise'
    ):
        super().__init__()
        self.predict = predict
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(4, d_model)

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

        # Output projection - predict noise for x, y, sin, cos
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 4)
        )

        # Layer norm for output
        self.norm_out = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

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
        h = self.input_proj(xysc)                           # B, N, D

        # 2. Add Color Embedding
        # Ensure colors are (B, N) for embedding lookup
        if colors.dim() == 3:
            colors = colors.squeeze(-1)                     # B, N

        h_color = self.color_embed(colors)                  # B, N, D
        h = h + h_color

        # 3. Add Time Embedding
        time_emb = self.time_embed(t).unsqueeze(1)
        h = h + time_emb

        # Add class embedding (broadcasted to all tiles)
        class_emb = self.class_embed(class_labels)          # B, class_embed_dim
        class_emb = self.class_proj(class_emb).unsqueeze(1) # B, 1, D
        h = h + class_emb

        # Apply transformer with unmasked self-attention
        h = self.transformer(h)                             # B, N, D

        # Normalize and project to noise
        h = self.norm_out(h)
        h = self.dropout(h)
        noise_pred = self.output_proj(h)                    # B, N, 4

        return noise_pred

class DDIMDiffuser(nn.Module):
    """DDIM diffusion process manager"""
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        β = torch.linspace(1e-5, 0.001, num_timesteps).view(-1, 1, 1)
        α = 1 - β
        ᾱ = torch.cumprod(α, dim=0)
        ᾱtm1 = torch.cat([torch.tensor([[[1.]]]), ᾱ[:-1]])

        self.register_buffer('β', β)
        self.register_buffer('α', α)
        self.register_buffer('ᾱ', ᾱ)
        self.register_buffer('ᾱtm1', ᾱtm1)
        self.register_buffer('rtᾱ', torch.sqrt(ᾱ))
        self.register_buffer('r1mᾱ', torch.sqrt(1. - ᾱ))

    def q_sample(self, xysc_0, t, noise=None):
        """
        Forward diffusion process:  xₜ = √ᾱₜ x₀ + √(1 − ᾱₜ) ε
        """
        if noise is None:
            noise = torch.randn_like(xysc_0)

        xysc_t = self.rtᾱ[t] * xysc_0 + self.r1mᾱ[t] * noise

        return xysc_t, noise

    def recover_xysc(self, xysc_t, t, noise):
        #  x̂₀ = (xₜ -  √(1 − ᾱₜ) ̂ϵ) / √ᾱₜ
        return (xysc_t - self.r1mᾱ[t] * noise) / self.rtᾱ[t]

    @torch.no_grad()
    def p_sample(self, denoiser, xysc_t, colors, t, class_labels, ddpm=0.0):
        """
        Reverse diffusion process (DDIM sampling)
        """
        if denoiser.predict == 'sample':
            xysc_0_pred = denoiser(xysc_t, colors, t, class_labels)
            noise_pred = (xysc_t - self.rtᾱ[t]*xysc_0_pred)/self.r1mᾱ[t]
            # ̂ϵ = (xₜ -  √ᾱₜ x̂₀) / √(1 − ᾱₜ)

        else:                                                       # predict noise
            noise_pred = denoiser(xysc_t, colors, t, class_labels)
            xysc_0_pred = (xysc_t - self.r1mᾱ[t]*noise_pred)/self.rtᾱ[t]
            #  x̂₀ = (xₜ -  √(1 − ᾱₜ) ̂ϵ) / √ᾱₜ


        if t[0] == 0:
            xysc_0_pred[..., 2:] = F.normalize(xysc_0_pred[..., 2:], dim=2)
            return xysc_0_pred

        # xₜ₋₁ = √(ᾱₜ₋₁) x̂₀ + √(1 − ᾱₜ₋₁ − σₜ²) ε_θ(xₜ) + σₜεₜ
        if ddpm == 0.:
            σₜ = 0.
            σₜεₜ = 0.

        else:
            σₜ = ddpm * torch.sqrt((1 - self.ᾱtm1[t]) / (1 - self.ᾱ[t]) * (1 - self.ᾱ[t] / self.ᾱtm1[t]))
            σₜεₜ = σₜ * torch.randn_like(xysc_t)

        xysc_new = torch.sqrt(self.ᾱtm1[t]) * xysc_0_pred + \
                   torch.sqrt(1 - self.ᾱtm1[t] - σₜ**2) * noise_pred + σₜεₜ

        # Circle Project
        # xysc_new[..., 2:] = F.normalize(xysc_new[..., 2:], dim=2)
        return xysc_new

    @torch.no_grad()
    def sample(self, denoiser, batch_size, num_tiles, class_labels, symmetry, num_steps=50, eta=0.0):
        """
        Generate samples using DDIM
        """
        device = next(denoiser.parameters()).device

        # Start from pure noise (only for first 4 dimensions)
        xysc = torch.randn((batch_size, num_tiles, 4), device=device)
        prob = {6: 1/3, 5: (3-5**0.5)/2}[symmetry]
        colors = (torch.rand((batch_size, num_tiles, 1), device=device) < prob).long()

        # Time steps for DDIM
        times = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=device).long()

        for i in range(num_steps):
            t = torch.full((batch_size,), times[i], device=device, dtype=torch.long) # type: ignore
            xysc = self.p_sample(denoiser, xysc, colors, t, class_labels, eta)
            maybe_mark_step()

        return xysc, colors
