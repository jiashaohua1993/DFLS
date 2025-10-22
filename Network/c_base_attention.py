
from einops import rearrange
from einops.layers.torch import Rearrange

import torch.nn as nn
import torch.nn.functional as F

import torch

class Downsample(nn.Module):
    def __init__(self, in_channels, inner_channels, times=2):
        """
        downsample the input feature map {times} times
        """
        super().__init__()
        self.downsample = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (p1 p2 c) h w",p1=times, p2=times),     # times is 4
            nn.Conv2d(in_channels * times * times, inner_channels, 1, 1, 0),   # 3*4*4   48, 256
        )

    def forward(self, x):
        assert x.dim() == 4, "The input tensor should be in 4 dims!"
        out = self.downsample(x)
        return out  # 5 256 128 128



class Upsample(nn.Module):
    def __init__(self, in_channels, inner_channels, up_times):
        """
        upsample {times} times
        """
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels * up_times * up_times, 1, 1, 0),
            Rearrange(
                "b (p1 p2 c) h w -> b c (h p1) (w p2)",
                c=inner_channels,
                p1=up_times,
                p2=up_times,
            ),
        )

    def forward(self, x):
        assert x.dim() == 4, "Input tensor should be in 4 dims!"
        out = self.upsample(x)
        return out


class FFN(nn.Module):
    def __init__(self, in_channels, inner_channels=None, dropout=0.0):
        super().__init__()
        inner_channels = in_channels if inner_channels is None else inner_channels
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, inner_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_channels, in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class LocalAttnLayer(nn.Module):
    def __init__(
        self, embedding_dim, patch_size=4, num_heads=3, dropout=0.0, shift_size=0):
        super().__init__()
        self.embedding = embedding_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.to_patches = nn.Sequential(
            Rearrange(
                "b c (hp p1) (wp p2) -> (p1 p2) (b hp wp) c",
                p1=patch_size,
                p2=patch_size,
            )
        )
        # multi-head attention
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads)
        # ffn
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = FFN(embedding_dim, embedding_dim * 4, dropout)

        self.shift_size = shift_size

        # pos encoding learnable
        self.row_embed = nn.Parameter(torch.Tensor(self.patch_size, self.embedding // 2))
        self.col_embed = nn.Parameter(torch.Tensor(self.patch_size, self.embedding // 2)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed)
        nn.init.uniform_(self.col_embed)

    def get_learnable_pos(self):
        pos = (torch.cat(
                [self.row_embed.unsqueeze(1).repeat(1, self.patch_size, 1),
                 self.col_embed.unsqueeze(0).repeat(self.patch_size, 1, 1),],
                dim=-1,).flatten(0, 1).unsqueeze(1))  # (p*p, 1, c)

        return pos  # (h*w, 1, c)


    def forward(self, x, attn_mask=None):
        B, C, H, W = x.shape

        # shift the non-overlapping patches
        if self.shift_size > 0:    #  seconde 2, when reconstruction
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # padding the feature maps
        padding_h = (self.patch_size - H % self.patch_size) % self.patch_size
        padding_w = (self.patch_size - W % self.patch_size) % self.patch_size

        x = F.pad(x, (0, padding_w, 0, padding_h))  # (l, r, t, b)
        h_paded, w_paded = x.shape[-2:]

        # split patches
        x_patches = self.to_patches(x)  # (p*p, b*h/p*w/p, c)  # 16*6400*256

        pos=self.get_learnable_pos()
        x_patches=x_patches+pos
        residual = x_patches

        # multi-head attention
        x_patches = self.norm1(x_patches)
        x_attn = self.attn(query=x_patches, key=x_patches, value=x_patches, attn_mask=attn_mask)[0]

        # residual
        x = x_attn + residual

        # feedforward network
        x = x + self.ffn(self.norm2(x))

        # patches to feature map
        x = rearrange(
            x,
            "(p1 p2) (b hp wp) c -> b c (hp p1) (wp p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            hp=h_paded // self.patch_size,
            wp=w_paded // self.patch_size,
        )
        if padding_h > 0 or padding_w > 0:
            x = x[..., :H, :W]
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        return x
