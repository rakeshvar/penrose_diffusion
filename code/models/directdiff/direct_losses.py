from abc import ABC

import torch
import torch.nn.functional as F

from ...utils.lossy import soft_assignment_matrix, get_lsa_indices
from ...utils.registry import Registry

loss_registry = Registry(name="DirectLoss")
register_loss = loss_registry.register

#------------------------------------------------------------------------------
# Abstract Base Class for Losses
#------------------------------------------------------------------------------
class AbstractLoss(ABC):
    def __init__(self, **kwargs):
        pass

    def __call__(self, xysc_0, xysc_t, noise, noise_hat, colors, t, diffuser=None):
        return self.compute_loss(xysc_0, xysc_t, noise, noise_hat, colors, t, diffuser=diffuser)

    def __repr__(self):
        return f"{self.__class__.__name__}"


#------------------------------------------------------------------------------
# NoisePredictionLoss / Flow Matching MSE Loss
#------------------------------------------------------------------------------
@register_loss('npl', 'noise')
class NoisePredictionLoss(AbstractLoss):
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t, diffuser=None):
        return F.mse_loss(noise, noise_hat)


#------------------------------------------------------------------------------
# Permutation Invariant Loss
#------------------------------------------------------------------------------
@register_loss('pil', 'pinvl', 'perminv', "pinv")
class PermutationInvariantLoss(AbstractLoss):
    """
    Loss calculation:
        xysc_0 is posterior estimated (combined) given observed xysc_t
        prior is that a tile in xysc_t can be from any true tile in xysc_0
        Equivalent target is calculated from xysc_0_posterior
    Doubly stochastic version of this is Sinkhorn Loss
    Hard version of this is LSA Serial
    """
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t, diffuser=None):
        assert diffuser is not None, "Diffuser/FlowMatcher is required for PermutationInvariantLoss"
        with torch.no_grad():
            σₓ, σₑ2 = diffuser.get_sigmas(t)
            P = soft_assignment_matrix(xysc_t, σₓ * xysc_0, colors, σₑ2, 'softmax')
            xysc_0_posterior = torch.bmm(P, xysc_0)
            target_equivalent = diffuser.recover_target(xysc_t, t, xysc_0_posterior)

        return F.mse_loss(noise_hat, target_equivalent)


#------------------------------------------------------------------------------
# Sinkhorn Doubly Stochastic Permutation Invariant Loss
#------------------------------------------------------------------------------
@register_loss("shl", "sinkhorn", "sink")
class SinkhornLoss(AbstractLoss):
    """
    This is almost same as Permutation Invariant Loss
    But this enforces that the Permutation Matrix is doubly stochastic
        All the columns and rows sum to 1
        Meaning all the truths are equally attended to
        So there is no danger of all the points collapsing to one
            while the rest of them are ignored
    """
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t, diffuser=None):
        assert diffuser is not None, "Diffuser/FlowMatcher is required for SinkhornLoss"
        with torch.no_grad():
            σₓ, σₑ2 = diffuser.get_sigmas(t)
            P = soft_assignment_matrix(xysc_t, σₓ * xysc_0, colors, σₑ2, 'sinkhorn')
            xysc0_posterior = torch.bmm(P, xysc_0)
            target_equivalent = diffuser.recover_target(xysc_t, t, xysc0_posterior)
        return F.mse_loss(noise_hat, target_equivalent)


#------------------------------------------------------------------------------
# Linear Sum Assignment Loss (Serial) CUDA/Scipy
#------------------------------------------------------------------------------
@register_loss('lsl', 'lsas', 'lsaserial', 'lpl', 'lsap', 'lsaparallel')
class LSALossSerial(AbstractLoss):
    """
    This is the most generous loss.
        We permute to recovered sample so that it is closest to the truth
        Then do MSE loss on target corresponding to that
    """
    def compute_loss(self, xysc_0, xysc_t, noise, noise_hat, colors, t, diffuser=None):
        assert diffuser is not None, "Diffuser/FlowMatcher is required for LSALoss"
        # Recover sample from predicted target
        with torch.no_grad():
            xysc_0_hat = diffuser.recover_x0(xysc_t, t, noise_hat)
            bi, ti, pi = get_lsa_indices(xysc_0_hat, xysc_0, colors)

        return F.mse_loss(noise[bi, ti], noise_hat[bi, pi])
