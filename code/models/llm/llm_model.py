import torch
import torch.nn as nn
import torch.nn.functional as F

from code.compatibility import maybe_mark_step
from code.models.base_model import AbstractModel
from code.utils.qrs import get_colors, qr_to_xya

class LLModel(AbstractModel):
    def __init__(self, config):
        super().__init__()
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
        return f"llm_d{self.d_model}x{self.n_layers}"

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
        logits = self._forward(qr, labels)                           # (B, 2N+1, V)

        # 3. Calculate Loss (Next Token Prediction)
        # We predict y given x.
        # Input sequence : [Class, q1, r1, ... qN, r(N-1)]
        # Target sequence: [q1,    r1, q2, ... rN]

        preds = logits[:, :-1, :]                                   # (B, 2N, V)
        qr_seq = qr.view(B, -1)
        targets = (qr_seq + self.vocab_size//2).clamp(0, self.vocab_size - 1)

        loss = F.cross_entropy(
            preds.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )

        aux_losses = torch.tensor([], device=self.device)
        return loss, aux_losses

    @torch.no_grad()
    def sample(self, colors, labels, num_steps=None):
        self.eval()
        B = labels.shape[0]
        N = self.num_tiles

        class_embs = self.class_embed(labels).unsqueeze(1) # (B, 1, D)
        curr_seq = class_embs + self.pos_embed[:, :1, :]
        generated_indices = []

        for i in range(N * 2):
            out = self.transformer(curr_seq, curr_seq)
            last_token_logits = self.out_head(out[:, -1, :])        # (B, V)

            if False:
                next_token = torch.argmax(last_token_logits, dim=-1) # Greedy
            else:
                probs = F.softmax(last_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(1) # Random sampling
            generated_indices.append(next_token)

            # Append and Embed for next step
            next_emb = self.qrs_embed(next_token).unsqueeze(1)
            # Add pos embedding
            next_emb = next_emb + self.pos_embed[:, i+1:i+2, :]

            curr_seq = torch.cat([curr_seq, next_emb], dim=1)
            maybe_mark_step()

        # 3. Reshape output to (B, N, 2)
        seq = torch.stack(generated_indices, dim=1).view(B, N, 2)

        # 4. Decode
        q = seq[..., 0] - self.vocab_size//2
        r = seq[..., 1] - self.vocab_size//2

        # 5. Convert to Continuous for visualization
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

