# pyright: reportIndexIssue=false
import torch
import torch.nn as nn
import math

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
        H = self.dim // 2
        i = torch.arange(H, device=device)
        ω = (1000 * math.pi) ** (-i/(H-1))          # Frequency ladder: ωᵢ = C^(-i/H)
        Φ = t[:, None] * ω[None, :]                 # Phase: Φ[b, i] = t[b] · ω[i]
        sinΦ, cosΦ = torch.sin(Φ), torch.cos(Φ)     # Embed
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
    ):
        super().__init__()

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
            noise: (B, N, 3) - predicted noise for [x, y, sin, cos]
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

class DDIMDiffusion(nn.Module):
    """DDIM diffusion process manager"""
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = torch.linspace(1e-5, 0.01, num_timesteps) 

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.]), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

    def q_sample(self, xysc_start, t, noise=None):
        """
        Forward diffusion process: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        """
        if noise is None:
            noise = torch.randn_like(xysc_start)

        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        xysc_noisy = a * xysc_start + b * noise

        return xysc_noisy, noise

    @torch.no_grad()
    def p_sample(self, denoiser, xysc, colors, t, class_labels, eta=0.0):
        """
        Reverse diffusion process (DDIM sampling)
        """
        # Predict noise
        noise_pred = denoiser(xysc, colors, t, class_labels)

        # Only denoise geometry (x, y, angle)
        xysc0_pred = (
            (xysc - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise_pred) /
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        )

        # DDIM update
        if t[0] > 0:
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1)
            alpha_t_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1)
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            noise = torch.randn_like(xysc) if eta > 0 else 0

            xysc_new = (
                torch.sqrt(alpha_t_prev) * xysc0_pred +
                torch.sqrt(1 - alpha_t_prev - sigma_t**2) * noise_pred +
                sigma_t * noise
            )
        else:
            xysc_new = xysc0_pred

        # Circle Project
        s = xysc_new[..., 2]
        c = xysc_new[..., 3]
        r = torch.sqrt(s*s + c*c + 1e-8)
        xysc_new[..., 2] = s / r
        xysc_new[..., 3] = c / r

        return xysc_new

    @torch.no_grad()
    def sample(self, denoiser, batch_size, mum_tiles_, class_labels, symmetry, num_steps=50, eta=0.0):
        """
        Generate samples using DDIM
        """
        device = next(denoiser.parameters()).device

        # Start from pure noise (only for first 4 dimensions)
        xysc = torch.randn((batch_size, mum_tiles_, 4), device=device)
        prob = {6: 1/3, 5: (3-5**0.5)/2}[symmetry]
        colors = (torch.rand((batch_size, mum_tiles_, 1), device=device) < prob).long()

        # Time steps for DDIM
        times = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=device).long()

        for i in range(num_steps):
            t = torch.full((batch_size,), times[i], device=device, dtype=torch.long) # type: ignore
            xysc = self.p_sample(denoiser, xysc, colors, t, class_labels, eta)

        return xysc, colors
