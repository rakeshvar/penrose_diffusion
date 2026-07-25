# pyright: reportIndexIssue=false
import torch
import torch.nn as nn
from code.utils.advanced import sample_ot_noise
from code.utils.lossy import gather_by_permutation, ot_cost_matrix
from code.utils.registry import Registry
from code.compatibility import maybe_mark_step

# Registry for different types of Diffusers/FlowMatchers
diffuser_registry = Registry("Diffuser")
register_diffuser = diffuser_registry.register

#------------------------------------------------------------------------------
# DDPM / DDIM Diffuser
#------------------------------------------------------------------------------
@register_diffuser('ddpm', 'ddim')
class Diffuser(nn.Module):
    """DDIM/DDPM diffusion process manager"""
    def __init__(self, ndims, num_timesteps=1000):
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
        self.register_buffer('ᾱ', ᾱ)
        self.register_buffer('onemᾱ', onemᾱ)
        self.register_buffer('rᾱ', torch.sqrt(ᾱ))
        self.register_buffer('r1mᾱ', torch.sqrt(1. - ᾱ))

        # Used for DDIM sampling
        self.register_buffer('ᾱtm1', ᾱtm1)

    def q_sample(self, x0, t, ϵ=None):
        """Forward diffusion process (adds noise)"""
        if ϵ is None:
            ϵ = torch.randn_like(x0)
        xₜ = self.rᾱ[t] * x0 + self.r1mᾱ[t] * ϵ
        return xₜ, ϵ

    def get_sigmas(self, t):
        """Returns (sigma_x, sigma_e_sq)"""
        return self.rᾱ[t], self.onemᾱ[t]

    def recover_target(self, xt, t, x0):
        """Recover target noise given xt and x0"""
        rᾱ = self.rᾱ[t]
        r1mᾱ = self.r1mᾱ[t]
        return (xt - rᾱ * x0) / r1mᾱ

    def recover_x0(self, xt, t, target_hat):
        """Recover x0 given xt and predicted noise"""
        rᾱ = self.rᾱ[t]
        r1mᾱ = self.r1mᾱ[t]
        return (xt - r1mᾱ * target_hat) / rᾱ

    @torch.no_grad()
    def p_sample(self, xₜ, ϵhat, t, ddpm=0.0, **kwargs):
        """
        Reverse diffusion process (DDIM/DDPM sampling)
        """
        x0hat = self.recover_x0(xₜ, t, ϵhat)

        if t[0] == 0:  # All t's are same in a prediction batch
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

    @torch.no_grad()
    def sample(self, denoiser, colors, labels, num_steps=50, ddpm=0.0, **kwargs):
        device = next(denoiser.parameters()).device
        B, N = colors.shape
        D = denoiser.io_dim
        x = torch.randn((B, N, D), device=device)
        times = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=device).long()

        for i in range(num_steps):
            t = torch.full((B,), times[i], device=device, dtype=torch.long)
            ϵhat = denoiser(x, colors, t, labels)
            x = self.p_sample(x, ϵhat, t, ddpm)
            maybe_mark_step()

        return x


#------------------------------------------------------------------------------
# Optimal Transport Flow Matcher
#------------------------------------------------------------------------------
@register_diffuser('flow')
class FlowMatcher(nn.Module):
    """Linear flow-matching process manager."""
    def __init__(self, ndims, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.ndims = ndims

    def q_sample(self, x0, t, ϵ=None):
        """Forward process: linear interpolation between x0 (data) and noise"""
        if ϵ is None:
            ϵ = torch.randn_like(x0)
        
        # s in [0, 1]. t=0 -> s=0 (data), t=T -> s=1 (noise)
        s = t.float() / (self.num_timesteps - 1)
        
        # Reshape s to match x0 dimensions
        s_view = s.view(-1, *([1] * self.ndims))
        
        # Clamp s to prevent exact 0 variance at s=0
        s_clamped = torch.clamp(s_view, min=1e-4)
        
        xt = (1. - s_clamped) * x0 + s_clamped * ϵ
        target = ϵ - x0
        return xt, target

    def get_sigmas(self, t):
        """Returns (sigma_x, sigma_e_sq)"""
        s = t.float() / (self.num_timesteps - 1)
        s_view = s.view(-1, *([1] * self.ndims))
        s_clamped = torch.clamp(s_view, min=1e-4)
        return (1. - s_clamped), s_clamped**2

    def recover_target(self, xt, t, x0):
        """Recover target velocity given xt and x0"""
        s = t.float() / (self.num_timesteps - 1)
        s_view = s.view(-1, *([1] * self.ndims))
        s_clamped = torch.clamp(s_view, min=1e-4)
        return (xt - (1. - s_clamped) * x0) / s_clamped

    def recover_x0(self, xt, t, target_hat):
        """Recover x0 given xt and predicted velocity"""
        s = t.float() / (self.num_timesteps - 1)
        s_view = s.view(-1, *([1] * self.ndims))
        s_clamped = torch.clamp(s_view, min=1e-4)
        return xt - s_clamped * target_hat

    @torch.no_grad()
    def p_sample(self, xt, target_hat, t, dt=1.0, **kwargs):
        """Reverse Flow Matching step (Euler integration)"""
        # ds is the step size in s-space
        ds = dt / (self.num_timesteps - 1)
        ds_view = ds.view(-1, *([1] * self.ndims))
        # x_{s - ds} = x_s - ds * target_hat
        return xt - ds_view * target_hat

    @torch.no_grad()
    def sample(self, denoiser, colors, labels, num_steps=50, **kwargs):
        device = next(denoiser.parameters()).device
        B, N = colors.shape
        D = denoiser.io_dim
        x = torch.randn((B, N, D), device=device)
        times = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_current = times[i]
            t_next = times[i+1]
            dt = t_current - t_next
            
            t = torch.full((B,), t_current.long(), device=device, dtype=torch.long)
            v_hat = denoiser(x, colors, t, labels)
            
            # Euler integration step
            dt_tensor = torch.full_like(t, dt, dtype=torch.float)
            x = self.p_sample(x, v_hat, t, dt=dt_tensor)
            maybe_mark_step()

        return x


@register_diffuser('otfm')
class OTFlowMatcher(FlowMatcher):
    """Tile-level OT flow matching with a structured three-dimensional base."""

    def q_sample(self, x0, t, ϵ=None, matcher=None):
        if x0.shape[-1] != 3:
            raise ValueError(f"OTFM expects scaled (x, y, angle), got {x0.shape}")
        if ϵ is None:
            ϵ = sample_ot_noise(x0.shape, x0.device, x0.dtype)
            if matcher is None:
                raise ValueError("OTFM requires matched noise or an assignment matcher")
            permutation = matcher.solve(ot_cost_matrix(x0, ϵ))
            ϵ = gather_by_permutation(ϵ, permutation)

        s = t.to(dtype=x0.dtype) / (self.num_timesteps - 1)
        s_view = s.view(-1, *([1] * self.ndims))
        xt = (1. - s_view) * x0 + s_view * ϵ
        return xt, ϵ - x0

    def get_sigmas(self, t):
        s = t.float() / (self.num_timesteps - 1)
        s_view = s.view(-1, *([1] * self.ndims))
        return (1. - s_view), s_view.square()

    def recover_target(self, xt, t, x0):
        s = t.to(dtype=xt.dtype) / (self.num_timesteps - 1)
        s_view = s.view(-1, *([1] * self.ndims))
        if torch.any(s_view == 0):
            raise ValueError("Velocity cannot be recovered from x_t at t=0")
        return (xt - (1. - s_view) * x0) / s_view

    def recover_x0(self, xt, t, target_hat):
        s = t.to(dtype=xt.dtype) / (self.num_timesteps - 1)
        s_view = s.view(-1, *([1] * self.ndims))
        return xt - s_view * target_hat

    @torch.no_grad()
    def sample(self, denoiser, colors, labels, num_steps=50, **kwargs):
        device = next(denoiser.parameters()).device
        B, N = colors.shape
        if denoiser.io_dim != 3:
            raise ValueError(f"OTFM denoiser must use io_dim=3, got {denoiser.io_dim}")
        x = sample_ot_noise((B, N, 3), device=device)
        times = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_current = times[i]
            t_next = times[i + 1]
            dt = t_current - t_next
            t = torch.full((B,), t_current.long(), device=device, dtype=torch.long)
            velocity = denoiser(x, colors, t, labels)
            dt_tensor = torch.full_like(t, dt, dtype=torch.float)
            x = self.p_sample(x, velocity, t, dt=dt_tensor)
            maybe_mark_step()
        return x
