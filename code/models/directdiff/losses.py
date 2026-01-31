from abc import ABC

import torch
import torch.nn.functional as F

from ..diffuser import Diffuser
from .direct_denoiser import TransformerDenoiser
from .isab_denoiser import ISABDenoiser
from ...utils_loss import circle_loss, equal_angle_loss_circular, lattice_loss, lsa_ordering_scipy, sinkhorn_permutation

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

def get_loss_functor_class(name: str):
    """Factory function to instantiate a loss by name (case-insensitive)."""
    key = name.lower()
    try:
        return _LOSS_REGISTRY[key]
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
                 denoiser: TransformerDenoiser | ISABDenoiser,
                 diffuser: Diffuser,
                 **kwargs_subclass):
        self.denoiser = denoiser
        self.diffuser = diffuser
        self.device = next(denoiser.parameters()).device

    def __call__(self, xysc_0, colors, labels):
        self.denoiser.train()
        B = xysc_0.shape[0]

        # Forward pass - Add Noise
        t = torch.randint(0, self.diffuser.num_timesteps, (B,), device=xysc_0.device).long()
        xysc_t, noise = self.diffuser.q_sample(xysc_0, t)

        # Predict noise/sample
        prediction = self.denoiser(xysc_t, colors, t.float(), labels)

        # Compute loss (subclass-specific)
        loss = self.compute_loss(xysc_0, xysc_t, noise, prediction, colors, t) # type: ignore

        return loss

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

class VPredictionLoss(AbstractLoss):
    """
    Assumes the denoiser predicts: v =  −√(1−αₜ) ⋅ x₀ + √αₜ ⋅ ε 
    This loss has stable scale across timesteps and does NOT require reweighting.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser.predict = 'v'

    def compute_loss(self, xysc_0, xysc_t, noise, v_hat, colors, t):
        v = self.diffuser.calculate_v(xysc_0, t, noise)
        return F.mse_loss(v, v_hat)

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
        xysc_0_hat = self.diffuser.recover_x(xysc_t, t, noise_hat)
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
    Doubly stocastic version of this is Sinkhorn Loss
    Hard version of this is LSA Parallel
    """
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        with torch.no_grad():
            # TODO: Scale xysc_0 by σₓ = sqrt(ᾱₜ)
            # TODO: use cost = a^2 + b^2 - 2ab
            diff = xysc_t.unsqueeze(2) - xysc_0.unsqueeze(1)     # B, N, N, D
            sq_dist = (diff ** 2).sum(dim=-1)                    # B, N, N
            noise_variance = 1.-self.diffuser.ᾱ[t]               # Var(noise) = 1-ᾱₜ # type: ignore
            logits = -sq_dist/(2.*noise_variance)

            color_mask = (colors.unsqueeze(2) == colors.unsqueeze(1)) # B, N, N
            logits[~color_mask] = -float("inf")
            soft_assignments = torch.softmax(logits, dim=-1)
            xysc_0_posterior = torch.bmm(soft_assignments, xysc_0)    # Batch matmul

            noise_target = self.diffuser.recover_ϵ(xysc_t, t, xysc_0_posterior)

        return F.mse_loss(noise_hat, noise_target)

#------------------------------------------------------------------------------
# Sinkhorn Doubly Stochastic Permutation Invariant Loss
#------------------------------------------------------------------------------
@register_loss("shl", "sinkhorn")
class SinkhornLoss(AbstractLoss):
    """
    This is almost same as Permutation Invariant Loss
    But this enfoces that the Permutation Matrix is doubly stochastic
        All the columns and rows sum to 1
        Meaning all the truths are equally attended to
        So there is no danger of all the points collapsing to one
            while the rest of them are ignored
    """
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        with torch.no_grad():
            σₓ = self.diffuser.rᾱ[t]          # B, 1, 1 # type: ignore
            σₑ = self.diffuser.r1mᾱ[t]         # B, 1, 1 # type: ignore
            twoσₑsqrd = 2.0 * (σₑ ** 2)        # B, 1, 1
            σₓxysc0 = σₓ * xysc_0              # B, N, 4

            # ---- squared distance ----
            a2 = (xysc_t ** 2).sum(dim=-1)[:, :, None]              # B, N, 1
            b2 = (σₓxysc0 ** 2).sum(dim=-1)[:, None, :]             # B, 1, N
            ab = torch.bmm(xysc_t, σₓxysc0.transpose(1, 2))         # B, N, N
            sq_dist = a2 + b2 - 2.0 * ab                            # B, N, N

            # ---- color constraint via cost masking ----
            diff_color = colors[:, :, None] != colors[:, None, :]   # B, N, N
            # scale BIG with σₑ² (critical for stability)
            BIG = 50. * twoσₑsqrd                                   # B, 1, 1
            sq_dist = sq_dist + diff_color * BIG

            # ---- Sinkhorn barycenters ----
            log_K = -sq_dist / twoσₑsqrd                             # B, N, N
            P = sinkhorn_permutation(log_K)                          # B, N, N
            σₓxysc0_post = torch.bmm(P, σₓxysc0)                     # B, N, 4

            # ---- noise target & loss ----
            noise_target = (xysc_t - σₓxysc0_post) / σₑ              # B, N, 4
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
        with torch.no_grad():
            xysc_0_hat = self.diffuser.recover_x(xysc_t, t, noise_hat)

            # Cost Matrix for LSA
            # TODO: use cost = a^2 + b^2 - 2ab
            diff = xysc_0_hat.unsqueeze(2) - xysc_0.unsqueeze(1)
            cost_matrix = (diff ** 2).sum(dim=-1)
            cost_np = cost_matrix.detach().cpu().numpy()
            colors_np = colors.detach().cpu().numpy()

            bi, ti, pi = lsa_ordering_scipy(cost_np, colors_np)
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
            # TODO: Need to scale xysc_0 by σₓ = sqrt(ᾱₜ)
            # TODO: Calculate cost = a^2 + b^2 - 2ab
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

        return loss