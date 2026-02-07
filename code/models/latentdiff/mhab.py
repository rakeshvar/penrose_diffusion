import torch.nn as nn


class MultiheadAttentionBlock(nn.Module):
    """
    Multihead Attention Block (MAB) from the Set Transformer paper.
    
    Performs cross-attention between X (queries) and Y (keys/values).
    - If X == Y → Self-Attention Block (SAB)
    - Includes:
        - Pre-attention LayerNorms (separate for Q and K/V, as in original impl)
        - Residual connection around attention
        - Post-attention LayerNorm
        - Residual Feed-Forward Network (FFN) with ReLU
    This exact structure matches the original Set Transformer (Lee et al., 2019)
    and is widely used in set/point cloud Transformers.
    """
    def __init__(self, dim, num_heads, ln=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.proj = nn.Linear(dim, dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim)
        )
        
        self.ln_q = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln_kv = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln_post_attn = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln_post_ff = nn.LayerNorm(dim) if ln else nn.Identity()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, a, b):
        a = self.ln_q(a)
        b = self.ln_kv(b)
        
        attn_out, _ = self.attn(a, b, b)  # a <- b
        attn_out = self.dropout(self.proj(attn_out))
        
        H = self.ln_post_attn(a + attn_out)  # Residual + norm
        
        # Feed-forward + residual
        ff_out = self.dropout(self.ff(H))
        out = self.ln_post_ff(H + ff_out)    # Residual + norm
        
        return out