
import torch.nn as nn
import torch
from einops import rearrange

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         # 编码器部分
#         self.enc1 = self.contract_block(3, 64)  # (batch_size, 64, 128, 128)
#         self.enc2 = self.contract_block(64, 128)  # (batch_size, 128, 64, 64)
#         self.enc3 = self.contract_block(128, 256)  # (batch_size, 256, 32, 32)
#         self.enc4 = self.contract_block(256, 512)  # (batch_size, 512, 16, 16)
#
#         # 中间层
#         self.middle = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.middle_bn = nn.BatchNorm2d(512)
#         self.middle_relu = nn.ReLU()
#
#         # 解码器部分
#         self.dec4 = self.expand_block(512, 256)  # (batch_size, 256, 32, 32)
#         self.dec3 = self.expand_block(256, 128)  # (batch_size, 128, 64, 64)
#         self.dec2 = self.expand_block(128, 64)  # (batch_size, 64, 128, 128)
#
#         # 输出层
#         self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # 输出通道为1
#
#     def contract_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#     def expand_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )
#
#     def forward(self, x):
#         # 编码器
#         enc1 = self.enc1(x)  # (batch_size, 64, 128, 128)
#         enc2 = self.enc2(enc1)  # (batch_size, 128, 64, 64)
#         enc3 = self.enc3(enc2)  # (batch_size, 256, 32, 32)
#         enc4 = self.enc4(enc3)  # (batch_size, 512, 16, 16)
#
#         # 中间层
#         middle = self.middle(enc4)  # (batch_size, 512, 16, 16)
#         middle = self.middle_bn(middle)
#         middle = self.middle_relu(middle)
#
#         # 解码器
#         dec4 = self.dec4(middle)  # (batch_size, 256, 32, 32)
#         dec3 = self.dec3(dec4)  # (batch_size, 128, 64, 64)
#         dec2 = self.dec2(dec3)  # (batch_size, 64, 128, 128)
#
#         # 输出层
#         output = self.final_conv(dec2)  # 输出 (batch_size, 1, 128, 128)
#
#         # 额外下采样到64x64
#         output = nn.functional.interpolate(output, size=(64, 64), mode='bilinear', align_corners=True)
#
#         output = torch.sigmoid(output)
#
#         return output.view(output.shape[0], -1)


class Discriminator(nn.Module):
    def __init__(
            self,
            dim=32,
            dim_mults=(1, 2, 4, 4, 8, 8, 16, 32),
            channels=3,
            with_time_emb=True,
    ):
        super().__init__()
        self.channels = dim

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        self.initial = nn.Conv2d(channels, dim, 1)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock_dis(dim_in, dim_out, norm=ind != 0),
                nn.AvgPool2d(2),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))) if ind >= (num_resolutions - 3) and not is_last else nn.Identity(),
                ConvNextBlock_dis(dim_out, dim_out),
            ]))
        dim_out = dim_mults[-1] * dim

        self.out = nn.Conv2d(dim_out, 1, 1, bias=False)


    def forward(self, x):
        x = self.initial(x)
        for convnext, downsample, convnext2 in self.downs:
            x = convnext(x)
            x = downsample(x)
            # x = attn(x)
            x = convnext2(x)
        return self.out(x).view(x.shape[0], -1)


class ConvNextBlock_dis(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim*2)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            nn.BatchNorm2d(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, 1, 1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            weight, bias = torch.split(condition, x.shape[1],dim=1)
            h = h * (1 + weight) + bias

        h = self.net(h)
        return h + self.res_conv(x)


def exists(x):
    return x is not None

from torchmetrics import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
def ssim_loss(pred, target):
    return 1 - ssim(pred, target)

