import torch
import torch.nn as nn
import torch.nn.functional as F

from code.augment import GeometryAugment
from code.utils_adv import pairwise_sq_dist

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
# Diffusion 
# ------------------------------------------------------------

class GaussianDiffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, z0, t=None, ε=None):
        if t is None:
            t = torch.randint_like(z0[:, 0], 0, self.T).long()
        if ε is None:
            ε = torch.randn_like(z0)
        a_bar = self.alpha_bar[t].unsqueeze(-1)
        zt = torch.sqrt(a_bar) * z0 + torch.sqrt(1 - a_bar) * ε
        return t, ε, zt

    @torch.no_grad()
    def sample_cfg(self,
        color,
        cls,
        guidance_scale=3.0
    ):
        B = color.shape[0]
        D = denoiser.time_embed.embedding_dim
        device = color.device
        NULL = denoiser.class_embed.num_embeddings - 1


        z = torch.randn(B, D, device=device)

        for t in reversed(range(diffusion.T)):
            t_ = torch.full((B,), t, device=device)

            # Denoiser
            eps_cond = denoiser(z, t_, cls)
            eps_uncond = denoiser(z, t_, torch.full_like(cls, NULL))
            eps = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
            z = (z - diffusion.betas[t] * eps) / torch.sqrt(diffusion.alphas[t])

        return decoder(z, color)



# ------------------------------------------------------------
# Latent Denoiser (CFG-capable)
# ------------------------------------------------------------

class LatentDenoiser(nn.Module):
    def __init__(self, D, num_classes, T=1000):
        super().__init__()
        self.time_embed = nn.Embedding(T, D)
        self.class_embed = nn.Embedding(num_classes + 1, D)

        self.net = nn.Sequential(
            nn.Linear(D, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, D),
        )

    def forward(self, z_t, t, cls):
        h = z_t + self.time_embed(t) + self.class_embed(cls)
        return self.net(h)


# ------------------------------------------------------------
# Reconstruction losses (Permutation invariant)
# ------------------------------------------------------------
def chamfer_loss(x, y, colors, t):
    dist = pairwise_sq_dist(x, y, colors, t)                # B, N, N
    loss_xy = dist.min(dim=2).values.mean()
    loss_yx = dist.min(dim=1).values.mean()
    return loss_xy + loss_yx


def sinkhorn_loss(x, y, colors, t):
    sq_dist = pairwise_sq_dist(x, y, colors, t)

    log_P = -sq_dist/(2*t)
    log_P = log_P - torch.logsumexp(log_P, dim=2, keepdim=True)  # rows
    log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)  # cols
    P = torch.exp(log_P)

    return (P * sq_dist).sum(dim=[1, 2]).mean()


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

#------------------------------------------------------------
# Model
# ------------------------------------------------------------

class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_tiles,
        num_classes,
        recr_loss,
        beta_kl=1e-3,
        p_uncond=.05,
        **ignore
    ):
        super().__init__()
        self.augmenter = GeometryAugment()
        self.encoder = SetEncoder(latent_dim, num_classes)
        self.denoiser = LatentDenoiser(latent_dim, num_classes)
        self.decoder = PerceiverDecoder(latent_dim, num_tiles)
        self.diffuser = GaussianDiffusion()
        self.recr_loss =  recr_loss
        self.beta_kl = beta_kl
        self.p_uncond = p_uncond
        self.null_class = num_classes
        self.latent_dim = latent_dim
    
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def descriptor(self):
        return f"ld{self.latent_dim}_{self.recr_loss}"

    def train_step(self, x, color, cls):
        # Encode to Latents
        mu, logvar = self.encoder(x, color, cls)
        loss_kl = kl_loss(mu, logvar)
        z0 = reparameterize(mu, logvar)

        # Latent Diffusion
        t, ε, zt = self.diffuser.q_sample(z0)

        # Drop some classes for Classifier Free Guidance
        cls_cond = cls.clone()
        drop = torch.rand_like(z0[:, 0]) < self.p_uncond
        cls_cond[drop] = self.null_class

        # Denoiser
        εhat = self.denoiser(zt, t, cls_cond)
        loss_diffusion = F.mse_loss(εhat, ε)

        # Decoder
        x_hat = self.decoder(z0, color)
        if self.recr_loss == "chamfer":
            loss_recr = chamfer_loss(x, x_hat, color, t)  
        elif self.recon_mode == "sinkhorn":
            loss_recr = sinkhorn_loss(x, x_hat, color, t)
        else:
            raise NotImplementedError(f"Unknown reconstruction loss: {self.recr_loss}")

        loss = loss_recr + self.beta_kl * loss_kl + loss_diffusion
        others = {
            "loss": loss.item(),
            "recr": loss_recr.item(),
            "klmv": loss_kl.item(),
            "diff": loss_diffusion.item(),
        }
        return loss

    
    def sample(self, labels, num_tiles, symmetry, num_steps):
        class_labels = torch.tensor(labels, device=self.device)

        xysc, colors = self.diffuser.sample(
            self.denoiser, 
            batch_size=class_labels.shape[0], 
            num_tiles=num_tiles,
            class_labels=class_labels, 
            symmetry=symmetry, 
            num_steps=num_steps
        )

        xyac = xysc_to_xyac(xysc, colors)

        return xyac
    



