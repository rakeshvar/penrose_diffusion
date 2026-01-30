import math
import torch
import torch.nn as nn

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
        i = torch.arange(n, device=device)
        ω = (1000 * math.pi) ** (-i/(n-1))          # Frequency: ωᵢ = 1 / (πT)ⁱ⁾ⁿ
        Φ = ω[None, :] * t[:, None]                 # Phase: Φ[m, i] = ω[i]·t[m] = tₘ / (πT)ⁱ⁾ⁿ
        sinΦ, cosΦ = torch.sin(Φ), torch.cos(Φ)     # sin(ωᵢt) = sin(t/(πT)ⁱ⁾ⁿ) , i ∈ [0, n)
        return torch.cat([sinΦ, cosΦ], dim=-1)
