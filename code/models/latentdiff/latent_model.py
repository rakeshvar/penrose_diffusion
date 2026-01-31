import torch
import torch.nn as nn
import torch.nn.functional as F

from code.augment import GeometryAugment
from code.models.latentdiff.latent_denoiser import LatentDenoiser
from code.utils_advanced import pairwise_sq_dist
from code.utils_loss import sinkhorn_permutation

from .set_decoder import PerceiverDecoder
from .set_encoder import SetEncoder

# ------------------------------------------------------------
# μ, σ2 → z
# ------------------------------------------------------------

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# ------------------------------------------------------------
# Reconstruction losses (Permutation invariant)
# ------------------------------------------------------------
def chamfer_loss(x, y, colors):
    sq_dist = pairwise_sq_dist(x, y, colors)                # B, N, N
    loss_xy = sq_dist.min(dim=2).values.mean()
    loss_yx = sq_dist.min(dim=1).values.mean()
    return (loss_xy + loss_yx)/2.


def sinkhorn_loss(x, y, colors):
    sq_dist = pairwise_sq_dist(x, y, colors)         # B, N, N   
    logits = -sq_dist/(2*.15**2)                     # .15 is unit_side of polygon
    P = sinkhorn_permutation(logits)
    y_post = torch.bmm(P, y)
    return F.mse_loss(x, y_post)


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
        D = denoiser.time_embed.embedding_dim
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

class LatentDiffusionModel(nn.Module):
    def __init__(self, config, num_tiles):
        super().__init__()

        latent_dim = config['latent_dim']
        num_classes = config['num_classes']
        rec_loss = config['loss']
        beta_kl = 1e-3
        p_uncond = 1/7.

        self.augmenter = GeometryAugment()
        self.encoder = SetEncoder(latent_dim, num_classes)
        self.denoiser = LatentDenoiser(latent_dim, num_classes)
        self.decoder = PerceiverDecoder(latent_dim, num_tiles)
        self.diffuser = LatentDiffuser(1)
        self.rec_loss =  rec_loss
        self.beta_kl = beta_kl
        self.p_uncond = p_uncond
        self.null_class = num_classes
        self.latent_dim = latent_dim
    
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def descriptor(self):
        return f"ld{self.latent_dim}_{self.rec_loss[:4]}"

    def train_step(self, x, color, cls):
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
        cls_cond[drop] = self.null_class

        # Denoiser
        εhat = self.denoiser(zt, t, cls_cond)                       # (B, D)
        loss_diffusion = F.mse_loss(εhat, ε)                        # (1,)

        # Decoder
        x_hat = self.decoder(z0, color)
        if self.rec_loss[:4] == "chamfer"[:4]:
            loss_recons = chamfer_loss(x, x_hat, color)  
        elif self.rec_loss[:4] == "sinkhorn"[:4]:
            loss_recons = sinkhorn_loss(x, x_hat, color)
        else:
            raise NotImplementedError(f"Unknown reconstruction loss: {self.rec_loss}")

        loss = loss_recons + self.beta_kl * loss_kl + loss_diffusion
        others = {
            "loss": loss.item(),
            "recr": loss_recons.item(),
            "klmv": loss_kl.item(),
            "diff": loss_diffusion.item(),
        }
        return loss

    
    def sample(self, colors, classes, num_steps):
        z = self.diffuser.sample(
            self.denoiser, 
            classes, 
            num_steps
        )

        xya = self.decoder(z, colors)
        xyac = torch.cat([xya, colors.unsqueeze(-1)], dim=-1)
        return xyac
