import torch

from code.compatibility import maybe_mark_step
from code.augment import GeometryAugment
from code.utils.advanced import (
    scaled_to_xyac,
    xya_to_scaled,
    xya_to_xysc,
    xysc_to_xyac,
)

from ..diffuser import diffuser_registry
from ..base_model import AbstractModel
from .direct_denoiser import TransformerDenoiser
from .isab_denoiser import ISABDenoiser
from .direct_losses import loss_registry
from .ot_prefetch import OTBatchPrefetcher, PreparedOTBatch


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
        self.is_otfm = diffuser_name == 'otfm'
        self.representation = model_config.get('representation', 'xysc')
        self.ot_prefetcher = None

        if self.is_otfm:
            if self.representation != 'scaled_xya':
                raise ValueError("OTFM requires representation='scaled_xya'")
            if model_config.get('io_dim') != 3:
                raise ValueError("OTFM requires io_dim=3")
            if model_config['loss'] not in ('npl', 'noise'):
                raise ValueError("OTFM requires plain velocity MSE loss ('npl')")

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

    def _train_from_endpoints(self, x0, noise, colors, cls):
        self.train()
        B = x0.shape[0]
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=x0.device).long()
        xt, target = self.diffuser.q_sample(x0, t, ϵ=noise)
        target_hat = self.denoiser(xt, colors, t.float(), cls)
        loss = self.loss_functor(
            x0,
            xt,
            target,
            target_hat,
            colors,
            t,
            diffuser=self.diffuser,
        )
        aux_losses = torch.tensor([], device=self.device)
        return loss, aux_losses

    def train_prepared_step(self, batch: PreparedOTBatch):
        if not self.is_otfm:
            raise RuntimeError("Prepared OT batches are only valid for OTFM")
        return self._train_from_endpoints(
            batch.x0,
            batch.noise,
            batch.colors,
            batch.labels,
        )

    def train_step(self, xya, colors, cls):
        if self.is_otfm:
            self._ensure_ot_prefetcher()
            pending = self.ot_prefetcher.prepare((xya, colors, cls))
            return self.train_prepared_step(self.ot_prefetcher.consume(pending))

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
        if self.is_otfm:
            scaled_xya, _ = xya_to_scaled(xya)
            t = torch.zeros_like(scaled_xya[:, 0, 0])
            velocity = self.denoiser(scaled_xya, colors, t, cls)
            x0 = self.diffuser.recover_x0(scaled_xya, t, velocity)
            return scaled_to_xyac(x0, colors)

        xysc, _ = xya_to_xysc(xya)
        t = torch.zeros_like(xysc[:, 0, 0])
        noise = self.denoiser(xysc, colors, t, cls)
        xyac = xysc_to_xyac(xysc - noise, colors)
        return xyac

    @property
    def aux_loss_names(self):
        return []

    def sample(self, colors, labels, num_steps):
        sample = self.diffuser.sample(
            self.denoiser,
            colors,
            labels,
            num_steps
        )

        if self.is_otfm:
            return scaled_to_xyac(sample, colors)
        return xysc_to_xyac(sample, colors)

    @property
    def uses_prepared_batches(self):
        return self.is_otfm

    def _ensure_ot_prefetcher(self):
        if self.ot_prefetcher is not None:
            return
        workers = self.config.get('ot_workers')
        workers = None if workers in (None, 'auto') else int(workers)
        self.ot_prefetcher = OTBatchPrefetcher(
            self.device,
            self.augmenter,
            max_workers=workers,
            seed=self.config.get('ot_seed'),
            async_enabled=self.config.get('ot_async_prefetch', True),
        )

    def iter_training_batches(self, loader):
        if not self.is_otfm:
            raise RuntimeError("Only OTFM uses prepared training batches")
        self._ensure_ot_prefetcher()
        return self.ot_prefetcher.iter_prepared(loader)

    @property
    def ot_mean_wait_ms(self):
        if self.ot_prefetcher is None:
            return 0.
        return self.ot_prefetcher.mean_wait_ms

    def runtime_setup(self, *args, **kwargs):
        if self.is_otfm:
            self._ensure_ot_prefetcher()

    def runtime_teardown(self):
        if self.ot_prefetcher is not None:
            self.ot_prefetcher.close()
            self.ot_prefetcher = None
