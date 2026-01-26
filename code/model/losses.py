from abc import ABC

import torch
import torch.nn.functional as F
import code.compatibility as compat

from code.model.ddim import DDIMDiffuser, TransformerDenoiser
from code.model.loss_helpers import circle_loss, equal_angle_loss_circular, lattice_loss, lsa_ordering_scipy, sinkhorn_permutation

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
                 device,
                 **kwargs_subclass):
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
@register_loss('npl', 'noise')
class NoisePredictionLoss(AbstractLoss):
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        return F.mse_loss(noise, noise_hat)

#------------------------------------------------------------------------------
# SamplePredictionLoss
#------------------------------------------------------------------------------
@register_loss('spl', 'sample')
class SamplePredictionLoss(AbstractLoss):
    """
    Properly scaled sample prediction loss (VP-consistent).
    Equivalent to noise MSE, but expressed in x0-space:
        xₜ = √ᾱₜ ̂x₀ + √(1 − ᾱₜ) ̂ε
        xₜ = √ᾱₜ x₀ + √(1 − ᾱₜ) ε
                ⇓
        (1 -ᾱₜ) ‖ϵ - ̂ϵ‖²   =   ᾱₜ ‖x₀ - ̂x₀‖²
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser.predict = 'sample'   # network predicts x0

    def compute_loss(self, xysc_0, xysc_t, noise, xysc_0_hat, colors, t):
        # t (hence weight) is different for each sample
        ᾱ_t = self.diffuser.ᾱ[t] # type: ignore
        loss = (xysc_0_hat - xysc_0).pow(2) * ᾱ_t / (1.-ᾱ_t)
        return loss.mean()

#------------------------------------------------------------------------------
# Assisted Loss
#------------------------------------------------------------------------------
@register_loss('nal', 'noiseassisted', 'assisted', 'asst')
class NoiseAssistedLoss(AbstractLoss):
    """
    We assist the training by encouraging
        - sin and cos be unity
        - all angles be equal       (valid for hex grids only)
        - tiles to be on a lattice  (valid for hex grids only)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unit_side = kwargs["side"] if "side" in kwargs else .18
        self.symmetry = kwargs["symmetry"] if "symmetry" in kwargs else 6

    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        xysc_0_hat = self.diffuser.recover_xysc(xysc_t, t, noise_hat)
        return F.mse_loss(noise, noise_hat) \
            + .33 * circle_loss(xysc_0_hat) \
            + .33 * equal_angle_loss_circular(xysc_0_hat) \
            + .33 * lattice_loss(xysc_0_hat, self.unit_side, self.symmetry)

#------------------------------------------------------------------------------
# Permutation Invariant Loss
#------------------------------------------------------------------------------
@register_loss('pil', 'pinvl', 'perminv')
class PermutationInvariantLoss(AbstractLoss):
    """
    Loss calculation:
        xysc_0 is posterior estimated (combined) given observed xysc_t
        prior is that a tile in xysc_t can be from any true tile in xysc_0
        Equivalent noise is calculated from xysc_0_posterior
    Hard version of this is LSA Parallel
    """
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        diff = xysc_t.unsqueeze(2) - xysc_0.unsqueeze(1)     # B, N, N, D
        sq_dist = (diff ** 2).sum(dim=-1)                    # B, N, N
        noise_variance = 1.-self.diffuser.ᾱ[t]               # Var(noise) = 1-ᾱₜ # type: ignore
        logits = -sq_dist/(2.*noise_variance)

        neginf = torch.tensor(-float("inf"), device=self.device)
        color_mask = (colors.unsqueeze(2) == colors.unsqueeze(1)) # B, N, N
        logits = torch.where(color_mask, logits, neginf)
        soft_assignments = torch.softmax(logits, dim=-1)
        xysc_0_posterior = torch.bmm(soft_assignments, xysc_0)    # Batch matmul

        noise_target = self.diffuser.recover_noise(xysc_t, t, xysc_0_posterior)
        return F.mse_loss(noise_hat, noise_target)

#------------------------------------------------------------------------------
# Sinkhorn Doubly Stochastic Permutation Invariant Loss
#------------------------------------------------------------------------------
@register_loss("shl", "sinkhorn")
class SinkhornLoss(AbstractLoss):
    """
    Sinkhorn-based permutation-invariant diffusion loss
    with color-constrained doubly-stochastic transport.
    This is almost same as Permutation Invariant Loss
    But this enfoces that the Permutation Matrix is doubly stochastic
        All the columns and rows sum to 1
        Meaning all the truths are equally attended to
        So there is no danger of all the points collapsing to one
            while the rest of them are ignored
    This is not fully differentiable as the posterior mean of truth
        is not differentiable with the current package
    """
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        B, N, D = xysc_t.shape

        σₓ = self.diffuser.rtᾱ[t]          # [B, 1, 1] # type: ignore
        σₑ = self.diffuser.r1mᾱ[t]         # [B, 1, 1] # type: ignore
        twoσₑsqrd = 2.0 * (σₑ ** 2)        # [B, 1, 1]

        # σₓ x₀
        σₓxysc0 = σₓ * xysc_0               # [B, N, 4]

        # ---- squared distance without [B,N,N,4] ----
        x2 = (xysc_t ** 2).sum(dim=-1)[:, :, None]              # [B, N, 1]
        y2 = (σₓxysc0 ** 2).sum(dim=-1)[:, None, :]             # [B, 1, N]
        xy = torch.bmm(xysc_t, σₓxysc0.transpose(1, 2))         # [B, N, N]
        cost = x2 + y2 - 2.0 * xy                               # [B, N, N]

        # ---- color constraint via cost masking ----
        diff_color = colors[:, :, None] != colors[:, None, :]   # [B, N, N]

        # scale BIG with σₑ² (critical for stability)
        BIG = 50.0 * twoσₑsqrd                                   # [B, 1, 1]
        cost = cost + diff_color * BIG

        # ---- Sinkhorn ----
        log_K = -cost / twoσₑsqrd                                # [B, N, N]
        P = sinkhorn_permutation(log_K)                          # [B, N, N]

        # ---- posterior mean ----
        σₓxysc0_post = torch.bmm(P, σₓxysc0)                     # [B, N, 4]

        # ---- noise target & loss ----
        noise_target = (xysc_t - σₓxysc0_post) / σₑ              # [B, N, 4]
        return F.mse_loss(noise_hat, noise_target)

#------------------------------------------------------------------------------
# Linear Sum Assignment Loss (Serial) CUDA/Scipy
#------------------------------------------------------------------------------
@register_loss('lsl', 'lsas', 'lsaserial')
class LSALossSerial(AbstractLoss):
    """
    This is the most generous loss.
        We permute to recovered sample so that it is closest to the truth
        Then do MSE loss on noise correspoding to that
    """
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        # Recover sample from predicted noise
        xysc_0_hat = self.diffuser.recover_xysc(xysc_t, t, noise_hat)

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

        return F.mse_loss(noise[bi, ti], noise_hat[bi, pi])

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
        Equivalently noise is permuted
        MSE (noise_permuted, noise_hat)
    Soft version of PermInvariantLoss
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
        loss = F.mse_loss(noise[bi, ti], noise_hat[bi, pi])

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        compat.optimizer_step(self.optimizer)

        return loss.item()