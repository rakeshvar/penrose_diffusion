import torch

from code.compatibility import maybe_mark_step
from code.augment import GeometryAugment
from code.utils.advanced import xya_to_xysc, xysc_to_xyac

from ..diffuser import diffuser_registry
from ..base_model import AbstractModel
from .direct_denoiser import TransformerDenoiser
from .isab_denoiser import ISABDenoiser
from .direct_losses import loss_registry


#------------------------------------------------------------------------------
# Diffusion Model
#------------------------------------------------------------------------------
class DirectDiffusionModel(AbstractModel):
    def __init__(self, model_config, dataset):
        super().__init__()
        self.config = model_config
        self.augmenter = GeometryAugment()

        # Independently select the diffuser/flowmatcher
        diffuser_name = model_config.get('diffuser', 'ddpm')
        DiffuserClass = diffuser_registry[diffuser_name]
        self.diffuser = DiffuserClass(ndims=2)

        # Independently select the denoiser
        if model_config['model'] == 'direct':
            self.denoiser = TransformerDenoiser(**model_config, num_classes=dataset.num_classes) # type: ignore
        elif model_config['model'] == 'isab':
            self.denoiser = ISABDenoiser(**model_config, num_classes=dataset.num_classes) # type: ignore
        else:
            raise NotImplementedError(f"Unknown model: {model_config['model']}")

        # Independently select the loss
        LossClass = loss_registry[model_config['loss']]
        self.loss_functor = LossClass()

    @property
    def descriptor(self):
        name = 'dir' if self.config['model'] == 'direct' else 'isa'
        return f"{name}{self.denoiser.d_model}x{self.config['num_layers']}_{self.loss_functor.abbr}"

    def train_step(self, xya, colors, cls):
        xya = self.augmenter(xya)
        xysc, _ = xya_to_xysc(xya)
        self.train()

        # Decoupled forward pass:
        B = xysc.shape[0]
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=xysc.device).long()
        xysc_t, target = self.diffuser.q_sample(xysc, t)

        # Predict target
        target_hat = self.denoiser(xysc_t, colors, t.float(), cls)

        # Compute loss
        loss = self.loss_functor(xysc, xysc_t, target, target_hat, colors, t, diffuser=self.diffuser)

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
