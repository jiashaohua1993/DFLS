from Network import a_multi_channel
from Network import b_fusion
import torch.nn as nn
from Network import d_net_utils

class FVSRs_en(nn.Module):
    def __init__(
        self,

        inner_channels=256,
        num_frames=5,
        patch_embedding_size=4,
        temporal_patchsize=4,
        temporal_two_layer=True,
        num_heads=8,
        dropout=0.0
    ):
        super().__init__()
        self.num_frames = num_frames
        self.down_times = patch_embedding_size
        self.feature_ex=a_multi_channel.Multi_channel_data_preprocess()



        self.temporal_fusion = b_fusion.TemporalFusion(
            inner_channels,
            num_frames,
            temporal_patchsize,
            num_heads,
            dropout,
            temporal_two_layer
        )


    def forward(self, x):
        assert x.dim() == 5, "Input tensor should be in 5 dims!"
        B, N, C, H, W = x.shape
        # image to featurs
        feats=self.feature_ex(x) # 15*256*64*64
        first_dim_size = feats.size(0)
        indices = [0]  # 总是提取第 0 维
        # 条件检查并添加索引
        if first_dim_size > 5:  # 如果第一个维度大于 5
            indices.append(5)

        if first_dim_size > 10:  # 如果第一个维度大于 10
            indices.append(10)
        # RB = feats[indices, :, :, :]
        feats = feats.reshape(B, N, -1, H // self.down_times, W // self.down_times) # 3 5 256 64 64
        #selected_feats = feats[:, 2, :, :, :]  # 选择第二维的索引为 2 的特征
        out_encoder = self.temporal_fusion(feats)  # 3 256 64 64
        out_encoder=out_encoder #+selected_feats
        return out_encoder

class FVSRs_de(nn.Module):
    def __init__(
            self,
            inner_channels=256,
            num_frames=5,
            patch_size=4,
            patch_embedding_size=4,
            num_layer_rec=20,
            num_heads=8,
            dropout=0.0,
            ffn_dim=None,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.down_times = patch_embedding_size
        self.reconstructor = nn.Sequential(
            d_net_utils.EncoderBlock(
                inner_channels,
                inner_channels,
                down_times=1,
                patch_size=patch_size,
                num_layers=num_layer_rec,
                num_heads=num_heads,
                dropout=dropout,
                ffn_dim=ffn_dim,
            )
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels * 4, 1, 1, 0),
            nn.PixelShuffle(2),
            nn.Conv2d(inner_channels, inner_channels * 4, 1, 1, 0),
            nn.PixelShuffle(2)
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(inner_channels, 3, 1, 1, 0))


    def forward(self,en_out,x):

#        en_out=rb+en_out
        out = self.reconstructor(en_out) # 3*5*256*64*64
        out = self.upsample(out)
        out = self.out_proj(out)
        #out_decoder = out+x[:, self.num_frames // 2]
        out_decoder = out + x[:, self.num_frames // 2]  #*0.01+x[:, 0]*0.99

        return out_decoder,out #,x[:, 0]


