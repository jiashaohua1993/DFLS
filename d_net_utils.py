
import torch.nn as nn
from Network import c_base_attention


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels,
        down_times=4,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        dropout=0.0,
        ffn_dim=None,
        attn_type="LocalAttnLayer",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.down_times = down_times
        self.patch_size = patch_size
        if down_times > 1:
            self.downsample = c_base_attention.Downsample(in_channels, inner_channels, down_times)

        self.attn_layers = nn.Sequential(
            *[
                c_base_attention.LocalAttnLayer(
                    self.inner_channels,
                    self.patch_size,
                    num_heads,
                    dropout,
                    shift_size=0 if i % 2 == 0 else self.patch_size // 2,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        """
        feats = x
        if self.down_times > 1:
            feats = self.downsample(feats)  # (b, c, h/d, w/d)
        return self.attn_layers(feats)