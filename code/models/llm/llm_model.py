import torch
import torch.nn as nn
import torch.nn.functional as F

from code.compatibility import maybe_mark_step
from code.models.base_model import AbstractModel

class LLModel(AbstractModel):
    def __init__(self, config, dataset):
        super().__init__()

        # Configuration
        self.config = config
        self.d_model = config['d_model']
        self.n_layers = config['num_layers']
        self.n_heads = config['num_heads']
        self.dropout = config['dropout']
        self.num_tiles = dataset.num_tiles
        self.num_classes = dataset.num_classes
        self.vocab_size = dataset.vocab_size
        self.canvas_xyac = dataset.canvas_xyac

        self.token_embed = nn.Embedding(self.vocab_size, self.d_model)
        self.class_embed = nn.Embedding(self.num_classes, self.d_model)

        self.max_seq_len = self.num_tiles + 1      # class_token + tile_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, self.d_model))

        # Transformer Encoder (GPT-style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True # Usually more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                 num_layers=self.n_layers)

        self.out_head = nn.Linear(self.d_model, self.vocab_size)

    @property
    def descriptor(self):
        return f"llm{self.d_model}x{self.n_layers}"
    
    @property
    def aux_loss_names(self):
        return ['accuracy']

    def _forward(self, tokens, labels):
        """
        Args:
            qr: (B, N, 2) Integers [-30, 30]
            labels: (B,) Integers [0, 69]
        """
        B, N = tokens.shape
        seq_len = N + 1

        token_emb = self.token_embed(tokens)                      # B, N, D

        class_embs = self.class_embed(labels).unsqueeze(1)       # B, 1, D

        # Concatenate: [Class, i1, i2, i3, ..., iN]
        embs = torch.cat([class_embs, token_emb], dim=1)         # B, N+1, D

        pos_emb = self.pos_embed[:, :seq_len, :]
        embs = embs + pos_emb
        neginf = torch.full((seq_len, seq_len), float('-inf'), device=embs.device)
        mask = torch.triu(neginf, diagonal=1)

        out = self.transformer(embs, mask=mask)       # B, N+1, D
        logits = self.out_head(out)                                 # B, N+1, V

        return logits

    def train_step(self, tokens, colors, labels):
        self.train()
        assert tokens.ndim == 2, "Expecting only one columns, did you pass xya?"

        logits = self._forward(tokens, labels)                      # B, N+1, V
        logits = logits[:, :-1, :]                                  # B, N, V
        loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),   # (B*N, V)
                    tokens.reshape(-1),                     # (B*N,)
                    reduction='mean'
            )
        
        # Accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            correct_tokens = preds == tokens
            acc = correct_tokens.float().mean().item()

        return loss, torch.tensor([acc], device=self.device)

    @torch.no_grad()
    def sample(self, colors, labels, num_steps=None):
        self.eval()

        B = labels.shape[0]
        N = self.num_tiles
        V = self.vocab_size
        D = self.d_model
        d = self.device

        # init to a `NULL` token, so that shapes don't change
        NULL = V
        generated = torch.full((B, N), NULL, dtype=torch.long, device=d)      # B, N
        embed = torch.zeros(B, N + 1, D, device=d)                            # B, N+1, D
        embed[:, 0, :] = self.class_embed(labels) + self.pos_embed[:, 0, :]
        mask = torch.full((N+1, N+1), float('-inf'), device=d)
        mask = torch.triu(mask, diagonal=1) # N+1, N+1
        forbid = torch.zeros(B, V + 1, dtype=torch.bool, device=d)        # B, V+1

        for i in range(N):
            L = i + 1

            out = self.transformer(embed, mask=mask)                          # B, N+1, D
            logits = self.out_head(out[:, i, :])                              # B, D → B, V

            # logits[b, g[b, :]] = -inf (we use forbid to handle V≡NULL)
            forbid.zero_()        
            forbid.scatter_(1, generated, True)
            logits = logits.masked_fill(forbid[:, :V], float('-inf'))         # B, V

            probs = F.softmax(logits, dim=-1)                                 # B, V
            next_tok = torch.multinomial(probs, 1).squeeze(1)                 # B
            generated[:, i] = next_tok                                        # B, N
            embed[:, L, :] = self.token_embed(next_tok) + self.pos_embed[:, L, :]

            maybe_mark_step()
            _ = next_tok[0].cpu().item()

        seq = generated                                                       # B, N
        xyac = self.canvas_xyac[seq.long()]                                   # B, N, 4
        return xyac

    def passthrough(self, tokens, colors, labels):
        self.eval()
        logits = self._forward(tokens, labels)                       # B, N+1, V
        logits = logits[:, :-1, :]                                   # B, N, V
        next_token = torch.argmax(logits, dim=-1)                    # B, N
        xyac = self.canvas_xyac[next_token.long()]
        return xyac