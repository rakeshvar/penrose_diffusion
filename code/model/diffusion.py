# pyright: reportIndexIssue=false
import torch
import torch.nn as nn

from code.compatibility import maybe_mark_step

"""
Variance Preserving Transformations for Diffusion 

⎡ xₜ ⎤ = ⎡  √αₜ        √(1−αₜ) ⎤ ⎡ x₀ ⎤
⎣ v  ⎦   ⎣ −√(1−αₜ)     √αₜ    ⎦ ⎣ ε  ⎦


⎡ x₀ ⎤ = ⎡  √αₜ       −√(1−αₜ) ⎤ ⎡ xₜ ⎤
⎣ ε  ⎦   ⎣  √(1−αₜ)     √αₜ    ⎦ ⎣ v  ⎦

x̂₀ = √αₜ · xₜ − √(1−αₜ) · v̂
̂ε  = √(1−αₜ) · xₜ + √αₜ · v̂

"""

class Diffuser(nn.Module):
    """DDIM diffusion process manager"""
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        β = torch.linspace(1e-5, 0.001, num_timesteps).view(-1, 1, 1)
        α = 1 - β
        ᾱ = torch.cumprod(α, dim=0)
        onemᾱ = 1 - ᾱ
        ᾱtm1 = torch.cat([torch.tensor([[[1.]]]), ᾱ[:-1]])

        # Variance and Standard Deviation
        # Move them to the right device for GPU/TPU
        self.register_buffer('ᾱ', ᾱ)
        self.register_buffer('onemᾱ', onemᾱ)
        self.register_buffer('rᾱ', torch.sqrt(ᾱ))
        self.register_buffer('r1mᾱ', torch.sqrt(1. - ᾱ))

        # Used for DDIM sampling
        self.register_buffer('ᾱtm1', ᾱtm1)

    def q_sample(self, x0, t, ϵ=None):
        if ϵ is None:
            ϵ = torch.randn_like(x0)

        # xₜ = √ᾱ[t] * x0 + √(1-ᾱ[t]) * ϵ
        xₜ = self.rᾱ[t] * x0 + self.r1mᾱ[t] * ϵ
        return xₜ, ϵ

    def recover_xysc(self, xₜ, t, ϵ):
        return (xₜ - self.r1mᾱ[t] * ϵ) / self.rᾱ[t]

    def recover_noise(self, xₜ, t, xysc_0):
        return (xₜ - self.rᾱ[t]*xysc_0) / self.r1mᾱ[t]
    
    def recover_noise_from_v(self, x_t, t, v_hat):
        return self.r1mᾱ[t] * x_t + self.rᾱ[t] * v_hat
    

    @torch.no_grad()
    def p_sample(self, denoiser, xₜ, colors, t, class_labels, ddpm=0.0):
        """
        Reverse diffusion process (DDIM sampling)
        """
        if denoiser.predict == 'sample':
            xysc_0_pred = denoiser(xₜ, colors, t, class_labels)
            ϵ_pred = (xₜ - self.rᾱ[t]*xysc_0_pred)/self.r1mᾱ[t]
            # ̂ϵ = (xₜ -  √ᾱₜ x̂₀) / √(1 − ᾱₜ)

        else:                                                       # predict ϵ
            ϵ_pred = denoiser(xₜ, colors, t, class_labels)
            xysc_0_pred = (xₜ - self.r1mᾱ[t]*ϵ_pred)/self.rᾱ[t]
            #  x̂₀ = (xₜ -  √(1 − ᾱₜ) ̂ϵ) / √ᾱₜ


        if t[0] == 0:               # All t's are same in a prediction batch
            # Circle Project
            # xysc_0_pred[..., 2:] = F.normalize(xysc_0_pred[..., 2:], dim=2)
            return xysc_0_pred

        # xₜ₋₁ = √(ᾱₜ₋₁) x̂₀ + √(1 − ᾱₜ₋₁ − σₜ²) ε_θ(xₜ) + σₜεₜ
        if ddpm == 0.:
            σₜ = 0.
            σₜεₜ = 0.

        else:
            σₜ = ddpm * torch.sqrt((1 - self.ᾱtm1[t]) / (1 - self.ᾱ[t]) * (1 - self.ᾱ[t] / self.ᾱtm1[t]))
            σₜεₜ = σₜ * torch.randn_like(xₜ)

        xysc_new = torch.sqrt(self.ᾱtm1[t]) * xysc_0_pred + \
                   torch.sqrt(1 - self.ᾱtm1[t] - σₜ**2) * ϵ_pred + σₜεₜ

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
