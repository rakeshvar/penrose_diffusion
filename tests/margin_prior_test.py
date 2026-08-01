import math
import unittest

import torch

from code.models.directdiff.margin_prior import AttentionPriorMargin
from code.models.directdiff.isab_denoiser import ISABDenoiser
from code.models.directdiff.direct_denoiser import TransformerDenoiser


def attention_residual(prior: AttentionPriorMargin, u):
    """
    The latent-noise residual map e(u) = u - μ(u) analyzed in the paper,
    written independently of the module's Σ/A computation so the test can
    compare the analytic Jacobian against autograd.
    u: (N, p) -> e: (N, p)
    """
    N, p = u.shape
    q, k = prior.Wq(u), prior.Wk(u)
    logits = q @ k.T / math.sqrt(p)
    logits = logits.masked_fill(torch.eye(N, dtype=torch.bool), float('-inf'))
    attn = logits.softmax(dim=-1)
    return u - attn @ u


class MarginPriorJacobianTest(unittest.TestCase):
    """det(I - Σ_i A) must equal det of the true diagonal Jacobian block ∂e_i/∂u_i."""

    def test_analytic_jacobian_matches_autograd(self):
        torch.manual_seed(0)
        N, p = 6, 3
        prior = AttentionPriorMargin(d_model=p, prior_dim=p).double()
        u = torch.randn(N, p, dtype=torch.double)

        J = torch.autograd.functional.jacobian(
            lambda inp: attention_residual(prior, inp), u
        )  # (N, p, N, p)

        sigma = prior.sigma(u.unsqueeze(0)).squeeze(0)          # N, p, p
        M = torch.eye(p, dtype=torch.double) - sigma @ prior.A  # N, p, p

        for i in range(N):
            torch.testing.assert_close(J[i, :, i, :], M[i], rtol=1e-8, atol=1e-8)

    def test_barrier_matches_logdet_of_true_jacobian(self):
        torch.manual_seed(1)
        N, p = 5, 3
        prior = AttentionPriorMargin(d_model=p, prior_dim=p).double()
        # Make value_proj identity so h == u and the residual map applies directly
        with torch.no_grad():
            prior.value_proj.weight.copy_(torch.eye(p))
        u = torch.randn(N, p, dtype=torch.double)

        J = torch.autograd.functional.jacobian(
            lambda inp: attention_residual(prior, inp), u
        )
        expected = torch.stack([
            -torch.linalg.slogdet(J[i, :, i, :])[1] for i in range(N)
        ])

        barrier = prior.barrier(u.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(barrier, expected, rtol=1e-8, atol=1e-8)


class MarginPriorTrainingTest(unittest.TestCase):
    def test_finite_loss_and_gradients(self):
        torch.manual_seed(2)
        prior = AttentionPriorMargin(d_model=32, prior_dim=8)
        h = torch.randn(2, 20, 32, requires_grad=True)

        loss = prior(h)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        for param in prior.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_singleton_set_returns_zero(self):
        prior = AttentionPriorMargin(d_model=16, prior_dim=4)
        h = torch.randn(3, 1, 16)
        self.assertEqual(prior(h).item(), 0.)

    def test_gradient_reaches_denoiser_embedding(self):
        """The margin penalty must shape the denoiser's embedding parameters."""
        torch.manual_seed(3)
        common = dict(
            num_classes=4, class_embed_dim=8, time_embed_dim=16,
            d_model=32, num_heads=4, num_layers=1, dropout=0., io_dim=4,
        )
        for Denoiser in (TransformerDenoiser, ISABDenoiser):
            denoiser = Denoiser(**common)
            prior = AttentionPriorMargin(d_model=32, prior_dim=8)

            xysc = torch.randn(2, 12, 4)
            colors = torch.randint(0, 2, (2, 12))
            t = torch.rand(2)
            cls = torch.randint(0, 4, (2,))

            h = denoiser.embed(xysc, colors, t, cls)
            self.assertEqual(h.shape, (2, 12, 32))

            prior(h).backward()
            grad = denoiser.input_proj.weight.grad
            self.assertIsNotNone(grad)
            self.assertGreater(grad.abs().sum().item(), 0.)

    def test_embed_consistent_with_forward(self):
        """ISAB forward must still run through the refactored embed()."""
        torch.manual_seed(4)
        denoiser = ISABDenoiser(
            num_classes=4, class_embed_dim=8, time_embed_dim=16,
            d_model=32, num_heads=4, num_layers=2, dropout=0., io_dim=4,
        )
        denoiser.eval()
        xysc = torch.randn(2, 12, 4)
        colors = torch.randint(0, 2, (2, 12))
        t = torch.rand(2)
        cls = torch.randint(0, 4, (2,))
        out = denoiser(xysc, colors, t, cls)
        self.assertEqual(out.shape, (2, 12, 4))


if __name__ == '__main__':
    unittest.main()
