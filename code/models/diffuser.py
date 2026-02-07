# pyright: reportIndexIssue=false
import torch
import torch.nn as nn

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
    def __init__(self, ndims, num_timesteps=1000): # Add ndims argument
        super().__init__()
        self.num_timesteps = num_timesteps

        # Define the view shape based on ndims
        # ndims=1 -> [-1, 1]       (Good for (B, D))
        # ndims=2 -> [-1, 1, 1]    (Good for (B, N, D) or images)
        view_shape = [-1] + [1] * ndims

        β = torch.linspace(1e-5, 0.001, num_timesteps).view(*view_shape)
        α = 1. - β
        ᾱ = torch.cumprod(α, dim=0)
        onemᾱ = 1. - ᾱ
        one = torch.tensor([1.]).view((1,) * (ndims + 1))
        ᾱtm1 = torch.cat([one, ᾱ[:-1]])

        # Variance and Standard Deviation
        # Move them to the right device for GPU/TPU
        self.register_buffer('ᾱ', ᾱ)
        self.register_buffer('onemᾱ', onemᾱ)
        self.register_buffer('rᾱ', torch.sqrt(ᾱ))
        self.register_buffer('r1mᾱ', torch.sqrt(1. - ᾱ))

        # Used for DDIM sampling
        self.register_buffer('ᾱtm1', ᾱtm1)

    def q_sample(self, x0, t, ϵ=None):
        if ϵ is None:   ϵ = torch.randn_like(x0)
        xₜ = self.rᾱ[t] * x0 + self.r1mᾱ[t] * ϵ
        return xₜ, ϵ

    def recover_x(self, xₜ, t, ϵ):
        return (xₜ - self.r1mᾱ[t] * ϵ) / self.rᾱ[t]

    def recover_ϵ(self, xₜ, t, x0):
        return (xₜ - self.rᾱ[t] * x0) / self.r1mᾱ[t]

    def calculate_v(self, x0, t, ϵ):
        return -self.r1mᾱ[t] * x0 + self.rᾱ[t] * ϵ


    @torch.no_grad()
    def p_sample(self, xₜ, ϵhat, t, ddpm=0.0):
        """
        Reverse diffusion process (DDIM sampling)
        """
        x0hat = self.recover_x(xₜ, t, ϵhat)


        if t[0] == 0:               # All t's are same in a prediction batch
            return x0hat

        # xₜ₋₁ = √(ᾱₜ₋₁) x̂₀ + √(1 − ᾱₜ₋₁ − σₜ²) ̂ε(xₜ) + σₜεₜ
        if ddpm == 0.:
            σₜ = 0.
            σₜεₜ = 0.
        else:
            σₜ = ddpm * torch.sqrt((1 - self.ᾱtm1[t]) / (1 - self.ᾱ[t]) * (1 - self.ᾱ[t] / self.ᾱtm1[t]))
            σₜεₜ = σₜ * torch.randn_like(xₜ)

        x_new = torch.sqrt(self.ᾱtm1[t]) * x0hat + \
                   torch.sqrt(1 - self.ᾱtm1[t] - σₜ**2) * ϵhat + σₜεₜ

        return x_new