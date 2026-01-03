# pyright: reportIndexIssue=false
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for TIME steps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        num_classes: int,     # 70
        class_embed_dim: int, # 128
        time_embed_dim: int,  # 256
        d_model: int,         # 256
        num_heads: int,       # 8
        num_layers: int,      # 6
        dropout: float,       # 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection - project point features to d_model
        self.input_proj = nn.Linear(4, d_model)
        
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
        
        # Output projection - predict noise for x, y, angle
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3)
        )
        
        # Layer norm for output
        self.norm_out = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, t, class_labels):
        """
        Args:
            x: (B, N, 4) - noisy points [x, y, angle, color]
            t: (B,) - time steps (normalized to [0, 1])
            class_labels: (B,) - class indices (0-69)
        Returns:
            noise: (B, N, 3) - predicted noise for [x, y, angle]
        """
        B, N, _ = x.shape
        
        # 1. Project input points
        h = self.input_proj(x)  # (B, N, d_model)
                
        # 3. Add time embedding (broadcasted to all points)
        time_emb = self.time_embed(t).unsqueeze(1)  # (B, 1, d_model)
        h = h + time_emb
        
        # 4. Add class embedding (broadcasted to all points)
        class_emb = self.class_embed(class_labels)  # (B, class_embed_dim)
        class_emb = self.class_proj(class_emb).unsqueeze(1)  # (B, 1, d_model)
        h = h + class_emb
        
        # 5. Apply transformer with unmasked self-attention
        h = self.transformer(h)  # (B, N, d_model)
        
        # 6. Normalize and project to noise
        h = self.norm_out(h)
        h = self.dropout(h)
        noise_pred = self.output_proj(h)  # (B, N, 3)
        
        return noise_pred

class DDIMDiffusion(nn.Module):
    """DDIM diffusion process manager"""
    def __init__(self, num_timesteps=1000, beta_schedule='linear'):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, num_timesteps)
        else:
            raise NotImplementedError
        
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
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start[..., :3])  # Only add noise to x, y, angle
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Only add noise to first 3 dimensions (x, y, angle)
        x_noisy = x_start.clone()
        x_noisy[..., :3] = (
            sqrt_alphas_cumprod_t * x_start[..., :3] +
            sqrt_one_minus_alphas_cumprod_t * noise
        )
        # Color stays unchanged
        x_noisy[..., 3:] = x_start[..., 3:]
        
        return x_noisy, noise
    
    @torch.no_grad()
    def p_sample(self, model, x, t, class_labels, eta=0.0):
        """
        Reverse diffusion process (DDIM sampling)
        """
        B, N, _ = x.shape
        
        # Predict noise
        noise_pred = model(x, t, class_labels)
        
        # Only denoise first 3 dimensions
        x0_pred = torch.zeros_like(x)
        x0_pred[..., :3] = (
            (x[..., :3] - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise_pred) /
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        )
        x0_pred[..., 3:] = x[..., 3:]  # Keep color unchanged
        
        # DDIM update
        if t[0] > 0:
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1)
            alpha_t_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1)
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            
            noise = torch.randn_like(x[..., :3]) if eta > 0 else 0
            
            # Update only first 3 dimensions
            x_new = torch.zeros_like(x)
            x_new[..., :3] = (
                torch.sqrt(alpha_t_prev) * x0_pred[..., :3] +
                torch.sqrt(1 - alpha_t_prev - sigma_t**2) * noise_pred +
                sigma_t * noise
            )
            x_new[..., 3:] = x[..., 3:]  # Keep color unchanged
        else:
            x_new = x0_pred
            
        return x_new
    
    @torch.no_grad()
    def sample(self, model, batch_size, num_polygons, class_labels, num_steps=50, eta=0.0):
        """
        Generate samples using DDIM
        """
        device = next(model.parameters()).device
        
        # Start from pure noise (only for first 3 dimensions)
        x = torch.randn((batch_size, num_polygons, 3), device=device)
        color = torch.randint(0, 2, (batch_size, num_polygons, 1), device=device).float()  # Binary color
        x = torch.cat([x, color], dim=-1)
        
        # Time steps for DDIM
        times = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=device).long()
        
        for i in range(num_steps):
            t = torch.full((batch_size,), times[i], device=device, dtype=torch.long) # type: ignore
            x = self.p_sample(model, x, t, class_labels, eta)
            
        return x
