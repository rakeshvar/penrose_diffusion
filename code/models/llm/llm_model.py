import torch
import torch.nn as nn
import torch.nn.functional as F

from code.compatibility import maybe_mark_step
from code.models.base_model import AbstractModel
from code.utils.qrs import get_colors, qr_to_xya

class LLModel(AbstractModel):
    def __init__(self, config):
        super().__init__()

        if config['symmetry'] == 5:
            raise NotImplementedError("LLM doesn't support 5-fold symmetry.")

        # 1. Configuration
        self.config = config
        self.num_tiles = config["num_tiles"]
        self.d_model = config['d_model']
        self.n_layers = config['num_layers']
        self.n_heads = config['num_heads']
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']
        self.vocab_size = 64

        self.qrs_embed = nn.Embedding(self.vocab_size, self.d_model)
        self.class_embed = nn.Embedding(self.num_classes, self.d_model)

        # Sequence length = (num_tiles * 2) + 1 (for class token)
        self.max_seq_len = (self.num_tiles * 2) + 1
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, self.d_model))

        # 4. Transformer Decoder (GPT-style)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True # Usually more stable
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)

        self.out_head = nn.Linear(self.d_model, self.vocab_size)

    @property
    def descriptor(self):
        return f"llm{self.d_model}x{self.n_layers}"

    @property
    def device(self):
        return next(self.parameters()).device

    def _forward(self, qr, labels):
        """
        Args:
            qr: (B, N, 2) Integers [-30, 30]
            labels: (B,) Integers [0, 69]
        """
        B, N, TWO = qr.shape
        seq_len = 2*N + 1

        qr_seq = qr.view(B, -1)                                  # (B, 2N)
        qr_tokens = (qr_seq + self.vocab_size//2).clamp(0, self.vocab_size - 1)
        qr_embs = self.qrs_embed(qr_tokens)                      # (B, 2N, D)

        class_embs = self.class_embed(labels).unsqueeze(1)       # (B, 1, D)

        # Concatenate: [Class, q1, r1, q2, r2, ...]
        cqr_embs = torch.cat([class_embs, qr_embs], dim=1)       # (B, 2N+1, D)

        pos_emb = self.pos_embed[:, :seq_len, :]
        cqr_embs = cqr_embs + pos_emb

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device) * float('-inf'),
            diagonal=1
        )

        out = self.transformer(cqr_embs, cqr_embs, tgt_mask=mask)   # (B, 2N+1, D)
        logits = self.out_head(out)                                 # (B, 2N+1, V)

        return logits

    def train_step(self, qr, colors, labels):
        self.train()
        B, N, TWO = qr.shape
        assert TWO == 2, "Expecting q and r as two columns, did you pass xya?"

        logits = self._forward(qr, labels)                           # (B, 2N+1, V)

        # 3. Calculate Loss (Next Token Prediction)
        # We predict y given x.
        # Input sequence : [Class, q1, r1, ... qN, r(N-1)]
        # Target sequence: [q1,    r1, q2, ... rN]

        preds = logits[:, :-1, :]                                   # (B, 2N, V)
        qr_seq = qr.view(B, -1)
        qr_seq = (qr_seq + self.vocab_size//2).clamp(0, self.vocab_size - 1)
        used_mask = self._build_used_pair_mask(qr)    # (B, 2N, V)
        preds = preds.masked_fill(used_mask, float('-inf'))
        loss = F.cross_entropy(
            preds.reshape(-1, self.vocab_size),
            qr_seq.reshape(-1)
        )

        aux_losses = torch.tensor([], device=self.device)
        return loss, aux_losses

    @torch.no_grad()
    def sample(self, colors, labels, num_steps=None):
        self.eval()
        B = labels.shape[0]
        N = self.num_tiles
        V = self.vocab_size
        device = self.device
        BR = torch.arange(B, device=device)

        # used_pairs[b, q, r] = True if (q,r) already used
        used_pairs = torch.zeros(B, V, V, device=device, dtype=torch.bool)  # (B, V, V)

        # start with class token (embedded + position 0)
        curr_seq = self.class_embed(labels).unsqueeze(1) + self.pos_embed[:, :1, :]  # (B, 1, D)

        generated = []

        for i in range(2 * N):
            out = self.transformer(curr_seq, curr_seq)   # (B, i, D)
            logits = self.out_head(out[:, -1, :])        # (B, V)

            # r-step masking (odd positions only)
            if i % 2 == 1:
                q_tok = generated[-1]                    # (B,)
                mask = used_pairs[BR, q_tok]             # (B, V)
                logits = logits.masked_fill(mask, float('-inf'))

            # sample
            probs = F.softmax(logits, dim=-1)             # (B, V)
            next_tok = torch.multinomial(probs, 1).squeeze(1)  # (B,)
            generated.append(next_tok)

            # update used_pairs after r-step
            if i % 2 == 1:
                r_tok = next_tok                          # (B,)
                used_pairs[BR, q_tok, r_tok] = True       # (B, V, V)  # type: ignore

            # append embedding + position
            next_emb = (
                self.qrs_embed(next_tok).unsqueeze(1)
                + self.pos_embed[:, i + 1 : i + 2, :]
            )                                             # (B, 1, D)

            curr_seq = torch.cat([curr_seq, next_emb], dim=1)
            maybe_mark_step()

        seq = torch.stack(generated, dim=1).view(B, N, 2)  # (B, N, 2)
        q = seq[..., 0] - V // 2
        r = seq[..., 1] - V // 2
        xya = qr_to_xya(q, r, 1.)
        colors = get_colors(q, r)
        return torch.cat([xya, colors.unsqueeze(-1)], dim=-1)

    def passthrough(self, qr, colors, labels):
        self.eval()
        B, N, TWO = qr.shape
        logits = self._forward(qr, labels)                           # (B, 2N+1, V)
        logits = logits[:, :-1, :]                                   # (B, 2N, V)

        next_qr = torch.argmax(logits, dim=-1)            # (B, 2N)
        next_qr = next_qr - self.vocab_size//2
        next_qr = next_qr.view(B, N, 2)
        q = next_qr[..., 0]
        r = next_qr[..., 1]

        xya = qr_to_xya(q, r, 1.)
        colors = get_colors(q, r)

        return torch.cat([xya, colors.unsqueeze(-1)], dim=-1)

    @property
    def aux_loss_names(self):
        return []



    def _build_used_pair_mask(self, qr):
        """
        qr: (B, N, 2)
        returns: (B, 2N, V)
        """
        B, N, _ = qr.shape
        V = self.vocab_size
        device = qr.device

        qr_tok = (qr + V // 2).clamp(0, V - 1)   # (B, N, 2)
        q = qr_tok[..., 0]                      # (B, N)
        r = qr_tok[..., 1]                      # (B, N)
        r_onehot = F.one_hot(r, V).bool()       # (B, N, V)
        used = torch.zeros(B, V, V, device=device, dtype=torch.bool)
        masks = []

        for i in range(N):
            masks.append(used[torch.arange(B), q[:, i]])
            used[torch.arange(B), q[:, i]] |= r_onehot[:, i]

        q_mask = torch.zeros(B, N, V, device=device, dtype=torch.bool)
        r_mask = torch.stack(masks, dim=1)      # (B, N, V)

        used_mask = torch.stack([q_mask, r_mask], dim=2)  # (B, N, 2, V)
        used_mask = used_mask.view(B, 2 * N, V)

        return used_mask
