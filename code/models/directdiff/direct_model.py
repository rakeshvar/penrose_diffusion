import torch

from code.compatibility import maybe_mark_step
from code.augment import GeometryAugment
from code.utils.advanced import xya_to_xysc, xysc_to_xyac

from ..diffuser import Diffuser
from ..base_model import AbstractModel
from .direct_denoiser import TransformerDenoiser
from .isab_denoiser import ISABDenoiser
from .direct_losses import loss_registry


#------------------------------------------------------------------------------
# Diffuser
#------------------------------------------------------------------------------
class DirectDiffuser(Diffuser):
    @torch.no_grad()
    def sample(self, denoiser, colors, labels, num_steps=50, ddpm=0.):
        device = next(denoiser.parameters()).device
        B, N = colors.shape
        D = denoiser.io_dim
        x = torch.randn((B, N, D), device=device)
        times = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=device).long()

        for i in range(num_steps):
            t = torch.full((B,), times[i], device=device, dtype=torch.long) # type: ignore
            ϵhat = denoiser(x, colors, t, labels)
            x = self.p_sample(x, ϵhat, t, ddpm)
            maybe_mark_step()

        return x

#------------------------------------------------------------------------------
# Diffusion Model
#------------------------------------------------------------------------------
class DirectDiffusionModel(AbstractModel):
    def __init__(self, model_config, dataset):
        super().__init__()
        self.config = model_config
        self.augmenter = GeometryAugment()
        self.diffuser = DirectDiffuser(2)

        if model_config['model'] == 'direct':
            self.denoiser = TransformerDenoiser(**model_config, num_classes=dataset.num_classes) # type: ignore
        elif model_config['model'] == 'isab':
            self.denoiser = ISABDenoiser(**model_config, num_classes=dataset.num_classes) # type: ignore
        else:
            raise NotImplementedError(f"Unknown model: {model_config['model']}")
        Loss = loss_registry[model_config['loss']]
        self.loss_functor = Loss(self.denoiser, self.diffuser)

    @property
    def descriptor(self):
        return f"{self.config['model'][0]}{self.denoiser.d_model}x{self.config['num_layers']}_{self.loss_functor.abbr}"

    def train_step(self, xya, colors, cls):
        xya = self.augmenter(xya)
        xysc, _ = xya_to_xysc(xya)
        self.train()
        loss = self.loss_functor(xysc, colors, cls)
        aux_losses = torch.tensor([], device=self.device)
        return loss, aux_losses

    def passthrough(self, xya, colors, cls):
        self.denoiser.eval()
        xysc, _ = xya_to_xysc(xya)
        t = torch.zeros_like(xysc[:, 0, 0])
        noise = self.denoiser(xysc, colors, t, cls)
        xyac = xysc_to_xyac(xysc - noise, colors)
        return xyac

    @property
    def aux_loss_names(self):
        return []

    def sample(self, colors, labels, num_steps):
        xysc = self.diffuser.sample(
            self.denoiser,
            colors,
            labels,
            num_steps
        )

        xyac = xysc_to_xyac(xysc, colors)
        return xyac