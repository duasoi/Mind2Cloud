import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
import math
from torch.nn import functional as F
import numpy as np
import sys
from mamba_ssm import Mamba2, Mamba
sys.path.append('.')
# import sys
# sys.path.append('/data2/hxf/neuro-3D-main/itrans_model/')
from eeg_data_process.clip_loss import ClipLoss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):  # 600 1500
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        # self.pe = pe
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x


class EEGAttention(nn.Module):  ### 时间维度上的attention
    def __init__(self, channel, d_model, nhead, max_len=600):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, src):
        # print("src shape", src.shape)  # ([128, 64, 600])  ([128, 64, 250])

        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        # print("after pos src shape", src.shape)  #  ([600, 128, 64])   ([250, 128, 64])

        output = self.transformer_encoder(src)
        # print("output shape", output.shape)  # 不变

        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]


class ConvBlock(nn.Module):
    def __init__(self, num_channels, num_features):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.norm3 = nn.LayerNorm(num_features)
        self.residual_conv = nn.Conv1d(num_channels, num_features, kernel_size=1)

    def forward(self, x):
        # print(f'ConvBlock input shape: {x.shape}')
        residual = self.residual_conv(x)
        # residual = x
        # print(f'residual shape: {residual.shape}')

        x = F.gelu(self.conv1(x))
        x = self.norm1(x)
        # print(f'After first convolution shape: {x.shape}')

        x = F.gelu(self.conv2(x))
        x = self.norm2(x)
        # print(f'After second convolution shape: {x.shape}')

        x = F.gelu(self.conv3(x))
        x = self.norm3(x)
        # print(f'After third convolution shape: {x.shape}')

        x += residual
        # print(f'ConvBlock output shape: {x.shape}')
        return x


class MLPHead(nn.Module):
    def __init__(self, in_features, num_latents, dropout_rate=0.25):
        super(MLPHead, self).__init__()

        self.layer1 = nn.Sequential(
            Rearrange('B C L->B L C'),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_latents),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            Rearrange('B L C->B (C L)'),
        )

    def forward(self, x):
        # print(f'MLPHead input shape: {x.shape}')
        x = self.layer1(x)
        # print(f'After first layer of MLPHead shape: {x.shape}')
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, Q, K, V):
        # Transformer 需要 (seq_len, batch_size, embed_dim)
        Q = Q.permute(2, 0, 1)
        K = K.permute(2, 0, 1)
        V = V.permute(2, 0, 1)

        attn_output, _ = self.multihead_attn(Q, K, V)

        attn_output = self.norm1(attn_output + Q)  # 残差连接

        ffn_output = self.ffn(attn_output)

        output = self.norm2(ffn_output + attn_output)  # 残差连接

        return output.permute(1, 0, 2)  # (seq_len, batch, embed_dim) → (batch, seq_len, embed_dim)

class CustomTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomTransformerLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, Q, K, V):
        # Q, K, V shape: (seq_length, batch_size, embed_dim)
        Q = Q.permute(2, 0, 1)
        K = K.permute(2, 0, 1)
        V = V.permute(2, 0, 1)

        attn_output, attn_output_weights = self.multihead_attn(Q, K, V)
        Q = self.norm1(Q + attn_output)

        ff_output = F.relu(self.linear1(Q))
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)
        output = self.norm2(Q + ff_output)

        return output.permute(1, 2, 0)


class VideoImageEEGClassifyColor3(nn.Module):
    def __init__(self, num_channels, sequence_length, sequence_length2, num_subjects=1, num_features=64,
                 num_latents=1024, num_blocks=1, cls_num=72):
        super(VideoImageEEGClassifyColor3, self).__init__()
        # default_config_dyn = Config()
        # default_config_dyn.seq_len = 600

        # default_config_stc = Config()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)

        self.static_attention = EEGAttention(num_channels, num_channels, nhead=1)

        # self.dyn_model = iTransformer(default_config_dyn)
        # self.stc_model = iTransformer(default_config_stc)

        self.static_linear = nn.Linear(sequence_length2, sequence_length2)
        self.dynamic_linear = nn.Linear(sequence_length, sequence_length2)  # # 600 250

        self.dynamic_static = CustomTransformerLayer(num_channels, num_heads=1)

        self.dynamic_static_1 = CrossAttentionLayer(num_channels, num_heads=1)

        self.conv_blocks = nn.Sequential(*[ConvBlock(num_channels, sequence_length2) for _ in range(num_blocks)],
                                         Rearrange('B C L->B L C'))

        self.mamba_dyn = Mamba2(
            d_model=sequence_length,
            d_state=64,
            d_conv=4,
            expand=8,
            headdim=24
        )

        self.mamba_stc = Mamba2(
            d_model=sequence_length2,
            d_state=16,
            d_conv=4,
            expand=4,
            headdim=25
        )

        self.linear_projection = nn.Sequential(
            Rearrange('B L C->B C L'),
            nn.Linear(sequence_length2, num_latents),
            Rearrange('B C L->B L C'))

        self.temporal_aggregation = nn.Linear(sequence_length2, 1)

        self.clip_head = MLPHead(num_latents, num_latents)
        self.class_head = nn.Linear(num_latents, 6)
        self.clip_head2 = MLPHead(num_latents, num_latents)
        self.class_head2 = nn.Linear(num_latents, cls_num)
        # self.mse_head = MLPHead(num_latents, num_latents)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self.loss_func = ClipLoss()

    def forward(self, x, x2):
        # import pdb;pdb.set_trace()
        # print("x x2", x.shape, x2.shape)  # x : ([128, 64, 600])  x2 : ([128, 64, 250])

        # x = x.permute(0, 2, 1)
        # x2 = x2.permute(0, 2, 1)

        # print("x shape", x.shape)
        # print("x2 shape", x2.shape)

        dyn = self.mamba_dyn(x)
        # print("after mamba2 dyn", dyn.shape)  # ([128, 64, 600])

        # dyn = dyn.permute(0, 2, 1)

        dyn = self.dynamic_linear(dyn)
        # print("after linear dyn", dyn.shape)


        stc = self.mamba_stc(x2)
        # print("after mamba stc", stc.shape)
        # stc = stc.permute(0, 2, 1)

        stc = self.static_linear(stc)
        # print("after linear stc", stc.shape) # ([128, 64, 250])

        x = self.dynamic_static(stc, dyn, dyn)  # class CustomTransformerLayer
        # print("x shape", x.shape) # ([128, 64, 250])

        # x = self.dynamic_static(dyn, stc, dyn)
        # stc_dyn = torch.cat([stc, dyn], dim=-1)
        # x = self.dynamic_static(stc_dyn)[..., 250:]

        x = self.conv_blocks(x)
        # print("x shape", x.shape) # ([128, 64, 250])

        x = self.linear_projection(x)
        # print(f'After linear projection shape: {x.shape}') # ([128, 1024, 250])

        # x_tem = self.temporal_aggregation(x)
        x_tem = torch.mean(x, dim=2, keepdim=True)
        # print(f'After temporal aggregation shape: {x_tem.shape}') # ([128, 1024, 1])

        clip_out = self.clip_head2(x_tem)  # class MLPHead
        # print("clip_out shape", clip_out.shape) # ([128, 1024])

        cls_result = self.class_head2(clip_out)
        # print("cls_result", cls_result.shape) # ([N, 72])  # shape / cls

        clip_out2 = self.clip_head(x_tem)  # class MLPHead
        # print("clip_out2 shape", clip_out2.shape) # ([128, 1024])

        cls_result2 = self.class_head(clip_out2)
        # print("cls_result2", cls_result2.shape) # ([N, 6])  # color

        return clip_out, cls_result, clip_out2, cls_result2

