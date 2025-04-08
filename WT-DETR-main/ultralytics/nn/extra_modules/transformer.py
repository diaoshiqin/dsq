import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv, autopad
from ..modules.transformer import TransformerEncoderLayer

__all__ = ['AIFI_LPE',]




class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, :self.seq_length]  # self.position_ids???????

        position_embeddings = self.pe(position_ids)
        return position_embeddings

class AIFI_LPE(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, fmap_size=20 * 20, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self.LPE = LearnedPositionalEncoding(fmap_size, c1, fmap_size)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.LPE(x)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


######################################## LearnedPositionalEncoding end ########################################

