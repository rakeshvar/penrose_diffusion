import torch
from code.augment import GeometryAugment
from code.models.base import DiffusionModel
from code.utils_adv import xya_to_xysc, xysc_to_xyac

from .denoiser import TransformerDenoiser
from ..diffuser import Diffuser
from .losses import get_loss_functor_class

class DirectDiffusionModel(DiffusionModel):
    def __init__(self, model_config):
        super().__init__()
        self.augmenter = GeometryAugment()
        self.diffuser = Diffuser(num_timesteps=1000)
        self.denoiser = TransformerDenoiser(**model_config) # type: ignore

        Loss = get_loss_functor_class(model_config['loss'])
        self.loss_functor = Loss(self.denoiser, self.diffuser)

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def descriptor(self):
        return f"d{self.denoiser.d_model}_{self.loss_functor.abbr}"
    
    def train_step(self, xya, colors, cls):
        xya = self.augmenter(xya)
        xysc, _ = xya_to_xysc(xya)
        loss = self.loss_functor(xysc, colors, cls)
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