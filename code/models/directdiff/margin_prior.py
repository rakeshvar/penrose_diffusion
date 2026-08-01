import math

import torch
import torch.nn as nn


class AttentionPriorMargin(nn.Module):
    """
    Set-attention port of the EmbeddingPrior from "Support Tokens, Stability
    Margins, and a New Foundation for Robust LLMs" (arXiv:2602.22271).

    The paper views attention as a latent-noise generator u_i = μ_i(u) + ε_i,
    whose exact log-density (by change of variables) contains the log-Jacobian
    of the residual map e_i = u_i - μ_i. Because the attention weights depend
    on u_i through the query, the diagonal Jacobian block is

        ∂e_i/∂u_i = I - Σ_i A,

    with Σ_i the attention-weighted covariance of the attended values and
    A = W_K^T W_Q / sqrt(p). The per-element barrier

        b_i = -log|det(I - Σ_i A)|

    diverges as element i approaches the degeneracy boundary det(I - Σ_i A)=0,
    so its mean acts as a smooth margin penalty (the paper's "margin-only"
    training variant).

    Differences from the causal original: elements attend over the whole set
    with the diagonal masked (the permutation-invariant analog of strict
    causality, which keeps ∂e_i/∂u_i exact), and embeddings are first projected
    to a small prior_dim so Σ_i stays (p x p) per element.
    """

    def __init__(self, d_model: int, prior_dim: int = 8):
        super().__init__()
        self.prior_dim = prior_dim
        self.value_proj = nn.Linear(d_model, prior_dim, bias=False)
        self.Wq = nn.Linear(prior_dim, prior_dim, bias=False)
        self.Wk = nn.Linear(prior_dim, prior_dim, bias=False)

    @property
    def A(self):
        # q_i = Wq.weight @ u_i and k_s = Wk.weight @ u_s, so A = W_K^T W_Q,
        # scaled by the same temperature used in the attention logits.
        return (self.Wk.weight.T @ self.Wq.weight) / math.sqrt(self.prior_dim)

    def sigma(self, u):
        """
        Attention-weighted covariance of attended values, diagonal masked.
        u: (B, N, p) -> Σ: (B, N, p, p)
        """
        _, N, p = u.shape
        q, k = self.Wq(u), self.Wk(u)
        logits = q @ k.transpose(-1, -2) / math.sqrt(p)         # B, N, N
        eye = torch.eye(N, dtype=torch.bool, device=u.device)
        logits = logits.masked_fill(eye, float('-inf'))
        attn = logits.softmax(dim=-1)                           # B, N, N

        u_bar = attn @ u                                        # B, N, p
        uu = torch.einsum('bsp,bsq->bspq', u, u)                # B, N, p, p
        second_moment = torch.einsum('bis,bspq->bipq', attn, uu)
        return second_moment - torch.einsum('bip,biq->bipq', u_bar, u_bar)

    def barrier(self, h):
        """
        Per-element barrier b_i = -log|det(I - Σ_i A)|.
        h: (B, N, d_model) -> (B, N)
        """
        B, N, _ = h.shape
        if N < 2:
            return h.new_zeros(B, N)

        u = self.value_proj(h)
        M = torch.eye(self.prior_dim, device=h.device, dtype=h.dtype) - self.sigma(u) @ self.A
        _, logabsdet = torch.linalg.slogdet(M)
        # Guard against -inf at exact singularity; gradient vanishes only there.
        return -logabsdet.clamp(min=-30.)

    def forward(self, h):
        return self.barrier(h).mean()
