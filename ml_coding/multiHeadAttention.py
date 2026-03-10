import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embd_dim: int, head_size: int, causal: bool = False, attn_dropout: float = 0.0):
        super().__init__()
        self.embd_dim = embd_dim
        self.head_size = head_size

        self.q = nn.Linear(embd_dim, head_size, bias=False)
        self.k = nn.Linear(embd_dim, head_size, bias=False)
        self.v = nn.Linear(embd_dim, head_size, bias=False)
        self.causal = causal

        # Scaled dot-product uses sqrt(d_k), where d_k = head_size.
        self.scale = head_size ** -0.5
        self.attn_dropout = nn.Dropout(p=attn_dropout)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, seq_len, embd_dim)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # scores shape: (batch, seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            t = x.size(1)
            mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # out shape: (batch, seq_len, head_size)
        out = weights @ v
        return out, weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embd_dim: int,
        head_size: int,
        causal: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if embd_dim % head_size != 0:
            raise ValueError(f"embd_dim ({embd_dim}) must be divisible by head_size ({head_size})")

        self.embd_dim = embd_dim
        self.head_size = head_size
        self.num_heads = embd_dim // head_size

        self.heads = nn.ModuleList(
            [
                Attention(
                    embd_dim=embd_dim,
                    head_size=head_size,
                    causal=causal,
                    attn_dropout=attn_dropout,
                )
                for _ in range(self.num_heads)
            ]
        )
        self.proj = nn.Linear(embd_dim, embd_dim)
        self.proj_dropout = nn.Dropout(p=proj_dropout)

    def forward(self, x: torch.Tensor):
        head_outputs = []
        attn_maps = []

        for head in self.heads:
            out, attn = head(x)
            head_outputs.append(out)
            attn_maps.append(attn)

        # concat heads on channel dim: (batch, seq_len, num_heads * head_size) = (batch, seq_len, embd_dim)
        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        out = self.proj_dropout(out)

        # stack maps: (batch, num_heads, seq_len, seq_len)
        attn_maps = torch.stack(attn_maps, dim=1)
        return out, attn_maps
