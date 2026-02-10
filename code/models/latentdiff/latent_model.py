from math import pi, sqrt
import torch
import torch.nn.functional as F

import code.compatibility as compat
from code.augment import GeometryAugment
from code.models.base_model import AbstractModel
from code.models.latentdiff.latent_denoiser import FiLMLatentDenoiser, MLPLatentDenoiser
from code.utils.lossy import lattice_loss

from .set_decoder import PerceiverDecoder
from .set_encoder import SetEncoder
from .latent_losses import loss_registry

# ------------------------------------------------------------
# μ, σ2 → z
# ------------------------------------------------------------

def reparameterize(mu, logvar):
    mu = torch.clamp(mu, min=-100.0, max=100.0)
    std = torch.exp(logvar/2.)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_loss(mu, logvar):
    return torch.mean(mu.pow(2) + logvar.exp() - 1 - logvar)/2.

#------------------------------------------------------------
# Diffuser
# ------------------------------------------------------------
from code.models.diffuser import Diffuser

class LatentDiffuser(Diffuser):
    @torch.no_grad()
    def sample(self, denoiser, labels, num_steps=50, guidance_scale=2.0):
        B = labels.shape[0]
        D = denoiser.dim
        device = denoiser.device
        NULL = denoiser.class_embed.num_embeddings - 1
        nulls = torch.full_like(labels, NULL)

        z = torch.randn((B, D), device=device)
        times = torch.linspace(self.num_timesteps-1, 0, num_steps+1, device=device).long()

        for i in range(num_steps):
            t = torch.full((B,), times[i], device=device, dtype=torch.long) # type: ignore
            ε_cond = denoiser(z, t, labels)
            ε_null = denoiser(z, t, nulls)
            ε_hatt = (1 + guidance_scale) * ε_cond - guidance_scale * ε_null
            z = self.p_sample(z, ε_hatt, t)

        return z


def check_tensor(name, x):
    if not torch.isfinite(x).all():
        raise RuntimeError(
            f"{name}: NaN/Inf "
            f"  min: {x.min().item()}"
            f"  avg: {x.mean().item()}"
            f"  max: {x.max().item()}"
        )

#------------------------------------------------------------
# Model
# ------------------------------------------------------------
LatentDenoiser = FiLMLatentDenoiser

class LatentDiffusionModel(AbstractModel):
    def __init__(self, config):
        super().__init__()

        L = config['latent_dim']
        C = config['num_classes']
        H = config['num_heads']
        P = config['num_pools']
        N = config['num_tiles']
        Kl = config['num_latent_blocks']
        Kv = config['num_vae_blocks']
        self.p_uncond = 1/7.
        self.config = config

        self.augmenter = GeometryAugment()
        self.encoder = SetEncoder(C, L, P, Kv, 2)
        self.denoiser = LatentDenoiser(C, L, Kl)
        self.decoder = PerceiverDecoder(N, L, Kv, H)
        self.diffuser = LatentDiffuser(1)
        self.recons_loss_fn = loss_registry[config['loss']]
        self.null_class = C

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def descriptor(self):
        D = self.config['latent_dim']
        K = self.config['num_latent_blocks']
        return f"lat{D}x{K}_{self.recons_loss_fn.abbr}"

    def train_step(self, x, color, cls):
        x = self.augmenter(x)
        x = x * torch.tensor([1., 1., sqrt(3)/pi], device=x.device)
        check_tensor("input x", x)

        self.train()
        B = x.shape[0]

        with compat.fp32():
            # Encode to Latents
            mu, logvar = self.encoder(x, color, cls)                    # mu: (B, D), logvar: (B, D)
            check_tensor("mu", mu)
            check_tensor("logvar", logvar)
            loss_kl = kl_loss(mu, logvar)                               # (1,)
            check_tensor("loss_kl", loss_kl)

            z0 = reparameterize(mu, logvar)                             # (B, D)
            check_tensor("z0", z0)

            # Latent Diffusion
            t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device) # (B,)
            zt, ε = self.diffuser.q_sample(z0, t)                       # zt: (B, D), ε: (B, D)
            check_tensor("zt", zt)
            check_tensor("ε", ε)

            # Drop some classes for Classifier Free Guidance
            cls_cond = cls.clone()
            drop = torch.rand_like(z0[:, 0]) < self.p_uncond
            nulls = torch.full_like(cls_cond, self.null_class)
            cls_cond = torch.where(drop, nulls, cls_cond)

            # Denoiser
            εhat = self.denoiser(zt, t, cls_cond)                       # (B, D)
            check_tensor("εhat", εhat)

            loss_diffusion = F.mse_loss(εhat, ε)                        # (1,)
            check_tensor("loss_diffusion", loss_diffusion)

            # Decoder
            x_hat = self.decoder(z0, color)
            check_tensor("x_hat", x_hat)

            loss_recons = self.recons_loss_fn(x, x_hat, color)
            check_tensor("loss_recons", loss_recons)

            # Lattice
            loss_lattice = lattice_loss(self.config['symmetry'], x_hat, self.config['side'])
            check_tensor("loss_lattice", loss_lattice)

            # Angle Variance
            loss_equiangle = torch.var(x_hat[:, :, 2], dim=1, unbiased=True).mean()
            check_tensor("loss_equiangle", loss_equiangle)

            loss = loss_recons
            if self.config["beta_kl"] > 0.:
                loss += self.config["beta_kl"] * loss_kl
            if self.config["beta_dl"] > 0.:
                loss += self.config["beta_dl"] * loss_diffusion
            if self.config["beta_ll"] > 0.:
                loss += self.config["beta_ll"] * loss_lattice

            check_tensor("final_loss", loss)

        aux_losses = torch.stack([
            loss_recons,
            loss_kl,
            loss_diffusion,
            loss_lattice,
            loss_equiangle
        ])
        return loss, aux_losses

    @property
    def aux_loss_names(self):
        return ['reconstruction', 'KL', 'Diffusion', 'Lattice', 'Equiangle']

    def passthrough(self, x, color, cls):
        self.eval()
        x = x * torch.tensor([1., 1., sqrt(3)/pi], device=x.device)
        mu, logvar = self.encoder(x, color, cls)                    # mu: (B, D), logvar: (B, D)
        z0 = reparameterize(mu, logvar)                             # (B, D)
        x_hat = self.decoder(z0, color)
        x_hat = x_hat * torch.tensor([1., 1., pi/sqrt(3)], device=x.device)
        xyac = torch.cat([x_hat, color.unsqueeze(-1)], dim=-1)
        return xyac

    def sample(self, colors, classes, num_steps):
        z = self.diffuser.sample(
            self.denoiser,
            classes,
            num_steps
        )
        self.eval()
        xya = self.decoder(z, colors)

        # rescale angle to [-π, π] and attach color
        xy, angle = xya.split([2, 1], dim=-1)
        angle = angle * (pi / sqrt(3))
        xyac = torch.cat([xy, angle, colors.unsqueeze(-1)], dim=-1)
        return xyac
