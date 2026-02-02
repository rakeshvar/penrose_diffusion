from abc import ABC

import torch
import torch.nn.functional as F

from code.utils.advanced import pairwise_sq_dist

from ..diffuser import Diffuser
from .direct_denoiser import TransformerDenoiser
from .isab_denoiser import ISABDenoiser
from ...utils.lossy import circle_loss_sincos, equiangle_loss_sincos, hex_lattice_loss_quadratic, lsa_ordering_scipy, sinkhorn_permutation

from ...utils.registry import Registry

loss_registry = Registry(name="DirectLoss")
register_loss = loss_registry.register

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


#------------------------------------------------------------------------------
# NoisePredictionLoss
#------------------------------------------------------------------------------
@register_loss('npl', 'noise')
class NoisePredictionLoss(AbstractLoss):
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t):
        return F.mse_loss(noise, noise_hat)

#------------------------------------------------------------------------------
# NoisePredictionLoss
#------------------------------------------------------------------------------
@register_loss('vpl', 'v')
class VPredictionLoss(AbstractLoss):
    """
    Assumes the denoiser predicts: v =  −√(1−αₜ) ⋅ x₀ + √αₜ ⋅ ε 
    This loss has stable scale across timesteps and does NOT require reweighting.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser.predict = 'v'         # to be used while sampling

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
        ᾱₜ = self.diffuser.ᾱ[t] # type: ignore
        loss = (xysc_0_hat - xysc_0).pow(2) * ᾱₜ / (1.-ᾱₜ)
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
            + .33 * circle_loss_sincos(xysc_0_hat) \
            + .33 * equiangle_loss_sincos(xysc_0_hat) \
            + .33 * hex_lattice_loss_quadratic(xysc_0_hat, self.unit_side, self.symmetry)

#------------------------------------------------------------------------------
# Permutation Invariant Loss
#------------------------------------------------------------------------------
@register_loss('pil', 'pinvl', 'perminv', "pinv")
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
            σₓ = self.diffuser.rᾱ[t]          # B, 1, 1 # type: ignore
            σₑ2 = self.diffuser.onemᾱ[t]                # type: ignore
            σₓxysc0 = σₓ * xysc_0             # B, N, 4

            sq_dist = pairwise_sq_dist(xysc_t, σₓxysc0, colors, σₑ2)
            logits = -sq_dist / (2*σₑ2)                             # B, N, N
            soft_assignments = torch.softmax(logits, dim=-1)
            xysc_0_posterior = torch.bmm(soft_assignments, xysc_0)  # B, N, 4

            noise_target = self.diffuser.recover_ϵ(xysc_t, t, xysc_0_posterior)

        return F.mse_loss(noise_hat, noise_target)

#------------------------------------------------------------------------------
# Sinkhorn Doubly Stochastic Permutation Invariant Loss
#------------------------------------------------------------------------------
@register_loss("shl", "sinkhorn", "sink")
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
            σₑ2 = self.diffuser.onemᾱ[t]                # type: ignore
            σₓxysc0 = σₓ * xysc_0             # B, N, 4

            sq_dist = pairwise_sq_dist(xysc_t, σₓxysc0, colors, σₑ2)
            logits = -sq_dist / (2*σₑ2)                              # B, N, N
            P = sinkhorn_permutation(logits)
            xysc0_posterior = torch.bmm(P, xysc_0)                   # B, N, 4

            noise_target = self.diffuser.recover_ϵ(xysc_t, t, xysc0_posterior)
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