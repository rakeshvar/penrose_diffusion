from abc import ABC

import torch
import torch.nn.functional as F
import code.compatibility as compat

from code.model.ddim import DDIMDiffuser, TransformerDenoiser
from code.model.loss_helpers import circle_loss, lsa_loss_cuda, lsa_ordering_scipy

#------------------------------------------------------------------------------
# Registry & Factory for Selecting Losses
#------------------------------------------------------------------------------
_LOSS_REGISTRY = {}

def register_loss(*aliases):
    """Decorator to register a loss class with multiple name aliases."""
    def decorator(cls):
        cls._canonical_name = aliases[0].lower() if aliases else cls.__name__.lower()
        for alias in aliases:
            _LOSS_REGISTRY[alias.lower()] = cls
        return cls
    return decorator

def get_loss(name: str, *args, **kwargs):
    """Factory function to instantiate a loss by name (case-insensitive)."""
    key = name.lower()
    try:
        return _LOSS_REGISTRY[key](*args, **kwargs)
    except KeyError:
        error_msg = f"Loss '{name}' not found. Available losses:\n"
        for cls_name, alias_list in list_losses().items():
            error_msg += f"  {cls_name}: {', '.join(alias_list)}\n"
        raise ValueError(error_msg)

def list_losses():
    """Returns a dict mapping loss class names to their aliases."""
    unique_classes = sorted(set(_LOSS_REGISTRY.values()), key=lambda x: x.__name__)
    return {cls.__name__: sorted([k for k, v in _LOSS_REGISTRY.items() if v == cls])
            for cls in unique_classes}

#------------------------------------------------------------------------------
# Abstract Base Class for Losses
#------------------------------------------------------------------------------
class AbstractLoss(ABC):
    def __init__(self,
                 denoiser: TransformerDenoiser,
                 diffuser: DDIMDiffuser,
                 optimizer,
                 device):
        self.denoiser = denoiser
        self.optimizer = optimizer
        self.diffuser = diffuser
        self.device = device

    def __call__(self, xysc_0, colors, labels):
        self.denoiser.train()
        B = xysc_0.shape[0]

        # Forward pass - Add Noise
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        xysc_t, noise = self.diffuser.q_sample(xysc_0, t)

        # Predict noise/sample
        prediction = self.denoiser(xysc_t, colors, t.float(), labels)

        # Compute loss (subclass-specific)
        loss = self.compute_loss(xysc_0, xysc_t, noise, prediction, colors, t) # type: ignore

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()

        # Universal Step (TPU/GPU)
        compat.optimizer_step(self.optimizer)

        return loss.item()

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @property
    def abbr(self):
        """Returns the canonical short name for this loss."""
        return getattr(self.__class__, '_canonical_name', self.__class__.__name__.lower())


#------------------------------------------------------------------------------
# NoisePredictionLoss
#------------------------------------------------------------------------------
@register_loss('npl', 'noise', 'n')
class NoisePredictionLoss(AbstractLoss):
    def compute_loss(self, xysc_0, xysc_t, noise, prediction, colors, t):
        # L2 Loss on noise prediction
        noise_hat = prediction
        return F.mse_loss(noise, noise_hat)

#------------------------------------------------------------------------------
# SamplePredictionLoss
#------------------------------------------------------------------------------
@register_loss('spl', 'sample', 'sp')
class SamplePredictionLoss(AbstractLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser.predict = 'sample'  # Important

    def compute_loss(self, xysc_0, xysc_t, noise, prediction, colors, t):
        # L2 Loss on sample prediction
        xysc_0_hat = prediction
        return F.mse_loss(xysc_0, xysc_0_hat)

#------------------------------------------------------------------------------
# SampleAssistedLoss
#------------------------------------------------------------------------------
@register_loss('sal', 'sa', 'sampleassisted', 'sampasst')
class SampleAssistedLoss(AbstractLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser.predict = 'sample'  # Important

    def compute_loss(self, xysc_0, xysc_t, noise, prediction, colors, t):
        xysc_0_hat = prediction
        return F.mse_loss(xysc_0, xysc_0_hat) + circle_loss(xysc_0_hat)

#------------------------------------------------------------------------------
# Linear Sum Assignment Loss (Serial) CUDA/Scipy
#------------------------------------------------------------------------------
@register_loss('lsl', 'lsas', 'lsaserial')
class LSALossSerial(AbstractLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        # Recover sample from predicted noise
        xysc_0_hat = self.diffuser.recover_xysc(xysc_t, t, noise_hat)

        if not compat.IS_TPU:
            return lsa_loss_cuda(xysc_0, xysc_0_hat, colors)

        else:
            # Cost Matrix for LSA
            diff = xysc_0_hat.unsqueeze(2) - xysc_0.unsqueeze(1)
            cost_matrix = (diff ** 2).sum(dim=-1)
            cost_np = cost_matrix.detach().cpu().numpy()
            colors_np = colors.detach().cpu().numpy()

            bi, ti, pi = lsa_ordering_scipy(cost_np, colors_np)
            with torch.no_grad():
                bi = torch.from_numpy(bi).to(self.device, non_blocking=True)
                ti = torch.from_numpy(ti).to(self.device, non_blocking=True)
                pi = torch.from_numpy(pi).to(self.device, non_blocking=True)

            return F.mse_loss(xysc_0[bi, ti], xysc_0_hat[bi, pi])

#------------------------------------------------------------------------------
# Linear Sum Assignment (Parallel) Scipy
#------------------------------------------------------------------------------
def maybe_stream(stream, enabled):
    from contextlib import nullcontext
    return torch.cuda.stream(stream) if enabled else nullcontext()

@register_loss('lpl', 'lsap', 'lsaparallel')
class LSALossParallel(AbstractLoss):
    """
    Loss calculation:
        xysc_0 is permuted so that it is closest to xysc_t
        MSE ( xysc_0_permuted, xysc_0_hat )
        Equivalently
        MSE (noise_permuted, noise_hat)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_cuda = (self.device.type == "cuda")
        self.stream_denoiser = torch.cuda.Stream() if self.use_cuda else None
        self.stream_cost = torch.cuda.Stream() if self.use_cuda else None

    def __call__(self, xysc_0, colors, labels):
        self.denoiser.train()
        B = xysc_0.shape[0]

        # Forward pass - Add Noise
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=self.device).long()
        xysc_t, noise = self.diffuser.q_sample(xysc_0, t)

        # Predict noise
        with maybe_stream(self.stream_denoiser, self.use_cuda):
            noise_hat = self.denoiser(xysc_t, colors, t.float(), labels)
            xysc_0_hat = self.diffuser.recover_xysc(xysc_t, t, noise_hat)

        # Cost Matrix for LSA
        with maybe_stream(self.stream_cost, self.use_cuda):
            diff = xysc_t.unsqueeze(2) - xysc_0.unsqueeze(1)
            cost_matrix = (diff ** 2).sum(dim=-1)

        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream_cost)

        cost_np = cost_matrix.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy()

        bi, ti, pi = lsa_ordering_scipy(cost_np, colors_np)

        with torch.no_grad():
            bi = torch.from_numpy(bi).to(self.device, non_blocking=True)
            ti = torch.from_numpy(ti).to(self.device, non_blocking=True)
            pi = torch.from_numpy(pi).to(self.device, non_blocking=True)

        # Ensure denoiser is done
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream_denoiser)

        # Loss
        loss = F.mse_loss(xysc_0[bi, ti], xysc_0_hat[bi, pi])

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        compat.optimizer_step(self.optimizer)

        return loss.item()