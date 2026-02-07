import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from code.compatibility import maybe_mark_step
from code.models.base_model import AbstractModel
from code.models.sinusoidal import SinusoidalPositionalEmbedding
from code.utils.qrs import get_colors, qr_to_xya

class MaskedDiscreteModel(AbstractModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Hyperparameters
        self.num_tiles = config['num_tiles']
        self.d_model = config['d_model']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']

        # Vocabulary
        # Range -32 to +31 centered at 32.
        # 0-63: Coordinate tokens
        # 64:   MASK TOKEN
        # 0 -> -31 , 1 -> -30 ⋅⋅⋅ 31 -> 0, 32 -> 1 ⋅⋅⋅ 62 -> 31, 63 -> MASK
        self.vocab_size = 63
        self.mask_token_id = self.vocab_size
        self.total_tokens = self.vocab_size + 1
        self.offset = 31

        # Embeddings
        self.coord_embed = nn.Embedding(self.total_tokens, self.d_model)
        self.class_embed = nn.Embedding(self.num_classes, self.d_model)

        # Positional Embedding (Learned is fine for fixed size grid)
        # Sequence length = 2 * num_tiles (q and r interleaved)
        self.seq_len = self.num_tiles * 2
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, self.d_model))

        # Time Embedding (Optional for MaskGIT, but helps model know 'how masked' it is)
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # Bidirectional Transformer (Encoder-only)
        # We use TransformerEncoder because it allows full visibility (non-causal)
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
    def device(self):
        return next(self.parameters()).device

    @property
    def aux_loss_names(self):
        return ['accuracy']

    def _forward(self, x_t, mask_ratio, labels):
        """
        x_t: (B, 2N) tokens (some masked)
        mask_ratio: (B,) float 0..1 indicating progress
        labels: (B,) class ids
        """
        B, L = x_t.shape
        h = self.coord_embed(x_t) # (B, 2N, D)
        h = h + self.pos_embed[:, :L, :]
        h = h + self.class_embed(labels).unsqueeze(1)
        # Map 0..1 ratio to 0..1000 timestep for sinusoidal
        t_int = (mask_ratio * 1000).long()
        t_emb = self.time_embed(t_int).unsqueeze(1)
        h = h + t_emb

        h = self.transformer(h)

        logits = self.out_head(h) # (B, 2N, V)
        return logits

    def train_step(self, qr, colors, labels):
        self.train()
        B, N, TWO = qr.shape
        def R(*a):  return torch.rand(*a, device=qr.device)

        # Tokeinze
        x_0 = qr.view(B, -1)                                # B, 2N
        x_0 = x_0 + self.offset    # clamp to vocab_size if runtime-error

        # Sample Masking Ratio = cos(uniform(0, pi/2))
        mask_ratio = torch.cos(R(B) * math.pi * 0.5)
        mask_bool = R(B, 2*N) < mask_ratio.unsqueeze(1)
        mask_token_tensor = torch.full_like(x_0, self.mask_token_id)
        x_t = torch.where(mask_bool, mask_token_tensor, x_0)

        logits = self._forward(x_t, mask_ratio, labels)   # B, 2N, V

        # Loss (Only on masked tokens)
        loss_all = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),          # 2BN, V
            x_0.reshape(-1),                              # 2BN
            reduction='none' # Return loss per token
        ).view(B, 2*N)
        loss_masked = loss_all * mask_bool.float()
        num_masked = mask_bool.sum().clamp(min=1.0)
        loss = loss_masked.sum() / num_masked

        # 6. Accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct_tokens = (preds == x_0) & mask_bool
            acc = correct_tokens.sum().float() / num_masked

        return loss, torch.tensor([acc], device=self.device)

    def passthrough(self, xya, colors, cls):
        return xya

    @torch.no_grad()
    def sample(self, colors, labels, num_steps=20):
        self.eval()
        B = labels.shape[0]
        N = self.num_tiles
        V = self.vocab_size
        d = self.device

        xₜ = torch.full((B, 2*N), self.mask_token_id, dtype=torch.long, device=d)      # B, 2N

        # Pre-compute duplicate detection mask (Lower Triangular)
        trl = torch.tril(torch.ones(N, N, device=d, dtype=torch.bool), diagonal=-1)    # N, N

        # Pre-compute S-Constraint Mask (|s| <= offset)
        a = torch.arange(V, device=d) - self.offset                                    # V
        qg, rg = torch.meshgrid(a, a, indexing='ij')                                   # V, V
        s_bad = ((qg + rg).abs() > self.offset).reshape(1, 1, -1)                      # 1, 1, V²

        for i in range(num_steps):
            t  = 1.0 - (i / num_steps)
            r  = math.sin(t * math.pi * 0.5)
            r1 = math.sin((1.0 - (i + 1)/num_steps) * math.pi * 0.5)
            rt = torch.full((B,), r, device=d)                                         # B

            # --- Joint Logits ---
            l = self._forward(xₜ, rt, labels)                                          # B, 2N, V
            ll = l.view(B, N, 2, V)                                                    # B, N, 2, V
            lj = ll[..., 0, :].unsqueeze(-1) + ll[..., 1, :].unsqueeze(-2)             # B, N, V, V
            lj = lj.view(B, N, -1)                                                     # B, N, V²

            # --- Forbid bad s values ---
            lj.masked_fill_(s_bad, float('-inf'))                                      # B, N, V²

            # --- Forbid unmasked pairs ---
            is_unmasked_pair = (xₜ != self.mask_token_id).view(B, N, 2).all(dim=-1)    # B, N
            xₜp = xₜ.view(B, N, 2)                                                     # B, N, 2
            ids = (xₜp[..., 0] * V + xₜp[..., 1])                                      # B, N
            forbidden = torch.zeros(B, V*V, device=d, dtype=torch.bool)                # B, V²
            for b in range(B):
                unmasked_pair_ids = ids[b][is_unmasked_pair[b]]
                forbidden[b, unmasked_pair_ids] = True
            lj.masked_fill_(forbidden.unsqueeze(1), float('-inf'))                     # B, N, V²
            #print where forbidden

            # --- Freeze unmasked pairs in place ---
            frozen = torch.full_like(lj, float('-inf'))                                # B, N, V²
            ids_ = ids.unsqueeze(-1).clamp(0, V*V - 1)                                 # B, N, 1
            frozen.scatter_(2, ids_, 1000.0)                # Overwrites at id_[b, i]    B, N, V²
            lj = torch.where(is_unmasked_pair.unsqueeze(-1), frozen, lj)               # B, N, V²

            # --- Sample ---
            pj = F.softmax(lj, dim=-1)                                                 # B, N, V²
            ids_new = torch.multinomial(pj.view(-1, V*V), 1).view(B, N)                # BN, V²  -> BN -> B, N
            xₜ_new = torch.stack([ids_new // V, ids_new % V], dim=2).view(B, 2*N)      # B, 2N

            # --- Get confidence ---
            cp = torch.gather(pj, 2, ids_new.unsqueeze(-1)).squeeze(-1)                # B, N
            confidence = cp.unsqueeze(-1).expand(-1, -1, 2).reshape(B, 2*N)            # B, 2N

            # --- Keep unmasked q, r ---
            confidence[xₜ != self.mask_token_id] = 1.                                  # B, 2N

            # --- Void Confidence for Duplicates ---
            matches = (ids_new.unsqueeze(-1) == ids_new.unsqueeze(1))                  # B, N, N
            is_dup = (matches & trl).any(dim=-1)                                       # B, N
            is_dup = is_dup.unsqueeze(-1).expand(-1, -1, 2).reshape(B, 2*N)            # B, 2N
            confidence[is_dup] = 0.                                                    # B, 2N

            # --- Break ties ---
            confidence += torch.rand_like(confidence) * 1e-4                           # B, 2N

            # --- Update State ---
            n_mask = int(r1 * 2 * N)

            # Nice Print Logic
            n_fixed = is_unmasked_pair.sum().item()
            n_dups = is_dup.sum().item() // 2
            if i > num_steps - 5 and n_dups > 0:
                print(f"Step {i+1:02d}/{num_steps} | Mask Rate: {r:.2%}→{r1:.2%} | Fixed Pairs: {n_fixed:3d} | New Dups: {n_dups:3d} | Target #Masked: {n_mask:3d}")

            if n_mask == 0:
                xₜ = xₜ_new # B, 2N
                break
            threshold = torch.kthvalue(confidence, n_mask, dim=1).values.unsqueeze(1)  # B, 1
            mask_decision = confidence < threshold                                     # B, 2N
            xₜ = torch.where(mask_decision, self.mask_token_id, xₜ_new)                # B, 2N
            maybe_mark_step()

        q = xₜ[:, 0::2] - self.offset # B, N
        r = xₜ[:, 1::2] - self.offset # B, N
        xya = qr_to_xya(q, r, 1.) # B, N, 2
        colors = get_colors(q, r).unsqueeze(-1) # B, N, 1
        return torch.cat([xya, colors], dim=-1) # B, N, 3