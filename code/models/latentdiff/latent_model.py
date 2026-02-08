from math import pi, sqrt
import torch
import torch.nn.functional as F

from code.augment import GeometryAugment
from code.models.base_model import AbstractModel
from code.models.latentdiff.latent_denoiser import LatentDenoiser
from code.utils.lossy import lattice_loss

from .set_decoder import PerceiverDecoder
from .set_encoder import SetEncoder
from .latent_losses import loss_registry

# ------------------------------------------------------------
# μ, σ2 → z
# ------------------------------------------------------------

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

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


#------------------------------------------------------------
# Model
# ------------------------------------------------------------

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

        self.train()
        B = x.shape[0]

        # Encode to Latents
        mu, logvar = self.encoder(x, color, cls)                    # mu: (B, D), logvar: (B, D)
        loss_kl = kl_loss(mu, logvar)                               # (1,)
        z0 = reparameterize(mu, logvar)                             # (B, D)

        # Latent Diffusion
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device) # (B,)
        zt, ε = self.diffuser.q_sample(z0, t)                       # zt: (B, D), ε: (B, D)

        # Drop some classes for Classifier Free Guidance
        cls_cond = cls.clone()
        drop = torch.rand_like(z0[:, 0]) < self.p_uncond
        nulls = torch.full_like(cls_cond, self.null_class)
        cls_cond = torch.where(drop, nulls, cls_cond)

        # Denoiser
        εhat = self.denoiser(zt, t, cls_cond)                       # (B, D)
        loss_diffusion = F.mse_loss(εhat, ε)                        # (1,)

        # Decoder
        x_hat = self.decoder(z0, color)
        loss_recons = self.recons_loss_fn(x, x_hat, color)

        # Lattice
        loss_lattice = lattice_loss(self.config['symmetry'], x_hat, self.config['side'])

        # Angle Variance
        loss_equiangle = torch.var(x_hat[:, :, 2], dim=1, unbiased=True).mean()

        loss = loss_recons
        if self.config["beta_kl"] > 0.:
            loss += self.config["beta_kl"] * loss_kl
        if self.config["beta_dl"] > 0.:
            loss += self.config["beta_dl"] * loss_diffusion
        if self.config["beta_ll"] > 0.:
            loss += self.config["beta_ll"] * loss_lattice

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
