import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from code.compatibility import maybe_mark_step
from code.models.base_model import AbstractModel
from code.models.sinusoidal import SinusoidalPositionalEmbedding

class MaskedDiscreteModel(AbstractModel):
    def __init__(self, config):
        super().__init__()

        # Hyperparameters
        self.config = config
        self.num_tiles = config['num_tiles']
        self.d_model = config['d_model']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']
        self.canvas_xyac = config['canvas_xyac']

        # Vocabulary
        assert config['vocab_size'] is not None, "Vocab size is None, dataset might be xya."
        self.vocab_size = config['vocab_size']
        self.mask_token_id = self.vocab_size
        self.total_tokens = self.vocab_size + 1

        # Embeddings
        self.coord_embed = nn.Embedding(self.total_tokens, self.d_model)
        self.class_embed = nn.Embedding(self.num_classes, self.d_model)

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tiles, self.d_model))

        # Time Embedding (Optional for MaskGIT, but helps model know 'how masked' it is)
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output Head
        self.out_head = nn.Linear(self.d_model, self.vocab_size) # No Mask

    @property
    def descriptor(self):
        return f"dis{self.d_model}x{self.num_layers}"

    @property
    def aux_loss_names(self):
        return ['accuracy']

    def _forward(self, x_t, t, labels):
        B, N = x_t.shape
        h = self.coord_embed(x_t)                               # B, N, D
        h = h + self.pos_embed[:, :N, :]
        h = h + self.class_embed(labels).unsqueeze(1)           #+B, 1, D
        
        # Map 0..1 ratio to 0..1000 timestep for sinusoidal
        t_int = (t * 1000).long()
        t_emb = self.time_embed(t_int).unsqueeze(1)             # B, 1, D
        h = h + t_emb

        h = self.transformer(h)

        logits = self.out_head(h)                               # B, N, V
        return logits

    def train_step(self, tokens, colors, labels):
        self.train()
        B, N = tokens.shape
        assert tokens.ndim == 2, "Expecting one column, did you pass xya?"
        def R(*a):  return torch.rand(*a, device=tokens.device)

        # Sample Masking Ratio = sin(uniform(0, pi/2))
        x_0 = tokens                                # B, N
        t = R(B)
        mask_ratio = torch.sin(t * math.pi * 0.5)
        mask_bool = R(B, N) < mask_ratio.unsqueeze(1)
        mask_token_tensor = torch.full_like(x_0, self.mask_token_id)
        x_t = torch.where(mask_bool, mask_token_tensor, x_0)

        logits = self._forward(x_t, t, labels)   # B, N, V

        # Loss (Only on masked tokens)
        loss_all = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),          # BN, V
            x_0.reshape(-1),                              # BN
            reduction='none' # Return loss per token
        ).view(B, N)
        loss_masked = loss_all * mask_bool.float()        # B, N
        num_masked = mask_bool.sum().float().clamp(min=1.0)
        loss = loss_masked.sum() / num_masked

        # Accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct_tokens = (preds == x_0) & mask_bool
            acc = correct_tokens.sum().float() / num_masked

        return loss, torch.tensor([acc], device=self.device)


    @torch.no_grad()
    def passthrough(self, tokens, colors, labels):
        self.eval()
        B, N = tokens.shape
        assert tokens.ndim() == 2, "Expecting one column, did you pass xya?"

        x_t = tokens                                      # B, N
        mask_ratio = torch.zeros(B, device=tokens.device)
        logits = self._forward(x_t, mask_ratio, labels)   # B, N, V

        preds = torch.argmax(logits, dim=-1)
        return preds
  

    @torch.no_grad()
    def sample(self, colors, labels, num_steps=20):
        self.eval()
        B = labels.shape[0]
        N = self.num_tiles
        V = self.vocab_size  # ≡ self.mask_token_id
        d = self.device

        # all mask
        xₜ = torch.full((B, N), V, dtype=torch.long, device=d)        # B, N

        # duplicate filter
        ones = torch.ones(N, N, device=d, dtype=torch.bool)           # N, N
        trl = torch.tril(ones, diagonal=-1)

        for i in range(num_steps):
            # schedule
            t  = 1. - i / num_steps
            r  = math.sin(t  * math.pi/2.)
            t1 = 1. - (i+1) / num_steps
            r1 = math.sin(t1 * math.pi/2.)
            tt = torch.full((B,), t, device=d)

            # forward
            logits = self._forward(xₜ, tt, labels)                      # B, N, V

            # forbid resampling already-fixed tokens
            unmasked = (xₜ != V)                     # B, N
            forbidden = torch.zeros(B, V+1, dtype=torch.bool, device=d) # B, V+1
            # f[b, xₜ[b, n]] = unm[b, n]
            forbidden.scatter_(1, xₜ, unmasked)
            forbidden = forbidden[:, :V].unsqueeze(1)                   # B, 1, V
            logits = logits.masked_fill(forbidden, float('-inf'))

            # sample
            probs = torch.softmax(logits, dim=-1)                       # B, N, V
            xₜ_new = torch.multinomial(probs.view(-1, V), 1).view(B, N) # B, N

            # freeze previous tokens
            xₜ_new = torch.where(unmasked, xₜ, xₜ_new)

            # confidence
            confidence = torch.gather(probs, 2, xₜ_new.unsqueeze(-1)).squeeze(-1)
            confidence = torch.where(unmasked, torch.ones_like(confidence), confidence)

            # duplicate removal
            matches = (xₜ_new.unsqueeze(-1) == xₜ_new.unsqueeze(1))
            is_dup = (matches & trl).any(dim=-1)
            confidence = confidence.masked_fill(is_dup, 0.0)

            # masking target (for next round)
            num_to_mask = int(r1 * N)

            # nice print diagnostics (last few steps only)
            if False and i > num_steps - 5:
                n_fixed = unmasked.sum().item()
                n_dups  = is_dup.sum().item()
                print(
                    f"Step {i+1:02d}/{num_steps} | "
                    f"Mask Rate: {r:.2%}→{r1:.2%} | "
                    f"Fixed: {n_fixed:3d} | "
                    f"New Dups: {n_dups:3d} | "
                    f"Target #Masked: {num_to_mask:3d}"
                )

            if num_to_mask == 0:
                xₜ = xₜ_new
                break

            # small noise to break ties
            confidence = confidence + torch.rand_like(confidence) * 1e-5
            # threshold
            threshold = torch.kthvalue(confidence, num_to_mask, dim=1).values.unsqueeze(1)
            # remask
            mask_next = confidence < threshold
            xₜ = torch.where(mask_next, self.mask_token_id, xₜ_new)

            maybe_mark_step()

        xyac = self.canvas_xyac[xₜ.long()]
        return xyac