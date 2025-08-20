import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
import math
from torch.nn import functional as F
import numpy as np
import sys
from torch import Tensor
from eeg_data_process.Embed import DataEmbedding, Encoder, EncoderLayer, AttentionLayer, FullAttention
sys.path.append('.')
# import sys
# sys.path.append('/data2/hxf/neuro-3D-main/itrans_model/')
from eeg_data_process.clip_loss import ClipLoss
# from itrans_model.itransformer import iTransformer


class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 600  # Sequence length
        self.pred_len = 600  # Prediction length
        self.output_attention = False  # Whether to output attention weights
        self.d_model = 250  # 192                  # Model dimension
        self.embed = 'timeF'  # Time encoding method
        self.freq = 'h'  # Time frequency
        self.dropout = 0.25  # Dropout rate
        self.factor = 1  # Attention scaling factor
        self.n_heads = 4  # Number of attention heads
        self.e_layers = 1  # Number of encoder layers
        self.d_ff = 256  # Dimension of the feedforward network
        self.activation = 'gelu'  # Activation function
        self.enc_in = 63  # Encoder input dimension (example value)




class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=None):
        super(iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, joint_train=False, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc):
        # Embedding
        # print(f'x_mark_enc:{x_mark_enc}') None
        # print(f'subject_ids: {subject_ids}') tensor([8, 8, 8,  ..., 8, 8, 8]
        # print(f'subject_ids shape: {subject_ids.shape}')  torch.Size([1024])

        # print(f'x_enc before embedding shape: {x_enc.shape}') torch.Size([1024, 63, 250])

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # print(f'enc_out after embedding shape: {enc_out.shape}') torch.Size([1024, 64, 250])

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print(f'enc_out after encoder shape: {enc_out.shape}') torch.Size([1024, 64, 250])
        enc_out = enc_out[:, :64, :]
        # print("enc_out", enc_out.shape)
        return enc_out


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)
        # print("x", x.shape)
        x = self.tsconv(x)
        # print("tsconv", x.shape)
        x = self.projection(x)
        # print("projection", x.shape)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=2880, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ATMS(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024,
                 num_blocks=1):
        super(ATMS, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x):
        x = self.encoder(x, None)
        # print("x shape", x.shape)


        # print(f'After attention shape: {x.shape}')
        # print("x", x.shape)
        # x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        # print("eeg_embedding shape", eeg_embedding.shape)

        out = self.proj_eeg(eeg_embedding)
        return out



# class EEGAttention(nn.Module):  ### 时间维度上的attention
#     def __init__(self, channel, d_model, nhead, max_len=600):
#         super(EEGAttention, self).__init__()
#         self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
#
#     def forward(self, src):
#         print("src shape", src.shape)  # ([128, 64, 600])  ([128, 64, 250])
#
#         src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
#         src = self.pos_encoder(src)
#         # print("after pos src shape", src.shape)  #  ([600, 128, 64])   ([250, 128, 64])
#
#         output = self.transformer_encoder(src)
#         # print("output shape", output.shape)  # 不变
#
#         return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]


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
        # default_config_dyn.pred_len = 600
        # default_config_dyn.d_model = 600

        # default_config_stc = Config()
        # self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)

        # self.static_attention = EEGAttention(num_channels, num_channels, nhead=1)

        self.dyn_model = ATMS()
        self.stc_model = ATMS()

        self.static_linear = nn.Linear(sequence_length2, sequence_length2)
        self.dynamic_linear = nn.Linear(sequence_length, sequence_length2)  # # 600 250

        self.dynamic_static = CustomTransformerLayer(num_channels, num_heads=1)

        self.conv_blocks = nn.Sequential(*[ConvBlock(num_channels, sequence_length2) for _ in range(num_blocks)],
                                         Rearrange('B C L->B L C'))

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

    def forward(self, x):
        # import pdb;pdb.set_trace()
        # print("x x2", x.shape, x2.shape)  # x : ([128, 64, 600])  x2 : ([128, 64, 250])

        # dyn = self.attention_model(x)  # class EEGAttention
        out = self.dyn_model(x)
        # print("after attention dyn", dyn.shape)  # ([128, 64, 600])

        # dyn = self.dynamic_linear(dyn)  # ([128, 64, 250])
        # print("after linear dyn", dyn.shape)

        # stc = self.static_attention(x2)  # class EEGAttention
        # stc = self.stc_model(x2)
        # print("after attention stc", stc.shape) # ([128, 64, 250])

        # stc = self.static_linear(stc)
        # print("after linear stc", stc.shape) # ([128, 64, 250])

        # x = self.dynamic_static(stc, dyn, dyn)  # class CustomTransformerLayer
        # print("x shape", x.shape) # ([128, 64, 250])

        # x = self.dynamic_static(dyn, stc, dyn)
        # stc_dyn = torch.cat([stc, dyn], dim=-1)
        # x = self.dynamic_static(stc_dyn)[..., 250:]

        # x = self.conv_blocks(x)
        # print("x shape", x.shape) # ([128, 64, 250])

        # x = self.linear_projection(x)
        # print(f'After linear projection shape: {x.shape}') # ([128, 1024, 250])

        # x_tem = self.temporal_aggregation(x)
        # x_tem = torch.mean(x, dim=2,  keepdim=True)
        # print(f'After temporal aggregation shape: {x_tem.shape}') # ([128, 1024, 1])

        # clip_out = self.clip_head2(x_tem)  # class MLPHead
        # print("clip_out shape", clip_out.shape) # ([128, 1024])

        # cls_result = self.class_head2(clip_out)
        # print("cls_result", cls_result.shape) # ([N, 72])  # shape / cls

        # clip_out2 = self.clip_head(x_tem)  # class MLPHead
        # print("clip_out2 shape", clip_out2.shape) # ([128, 1024])

        # cls_result2 = self.class_head(clip_out2)
        # print("cls_result2", cls_result2.shape) # ([N, 6])  # color

        # return clip_out, cls_result, clip_out2, cls_result2
        return out

