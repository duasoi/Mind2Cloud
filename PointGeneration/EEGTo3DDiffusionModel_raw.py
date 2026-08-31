import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers import ModelMixin
from tqdm import tqdm
import torch.nn as nn

from .pvn_model import PVNModel
from einops.layers.torch import Rearrange
import numpy as np
import sys
import os
# from chamferdist import ChamferDistance
sys.path.append('.')

from eeg_data_process.GAN import EnhancedPointCloudDiscriminator

def get_custom_betas(beta_start: float, beta_end: float, warmup_frac: float = 0.3, num_train_timesteps: int = 1000):
    """Custom beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_frac = 0.3
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas


class EEGTo3DDiffusionModel(ModelMixin):

    def __init__(
            self,
            beta_start: float,  # 1e-5
            beta_end: float,  # 8e-3
            beta_schedule: str,  # linear
            point_cloud_model_embed_dim: int,  # 64
            in_channels: int,  # 1027
            out_channels=3,
            sub='sub13',
            classifier=None,
            generate_type='color',  # shape or color
            retri_pretrain_model='',  # cls_model
            **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == 'custom':
            scheduler_kwargs.update(dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end)))
        else:
            scheduler_kwargs.update(dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            'pndm': PNDMScheduler(**scheduler_kwargs),
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference
        self.in_channels = in_channels  # 1027
        self.out_channels = out_channels  # 3
        self.video_clip_mlp = nn.Linear(1024, 64)

        # self.classifier = get_model(72, normal_channel=False)
        self.classifier = classifier

        self.use_aux_cls = False

        self.use_gan = True
        if self.use_gan:
            self.discriminator = EnhancedPointCloudDiscriminator(input_dim=3)
            self.gan_loss = nn.BCEWithLogitsLoss()

        # self.point_clip_mlp = nn.Linear(768, 64)
        # self.generate_type = generate_type  # shape


        self.mse_loss_fn = nn.MSELoss()
        # self.cd_loss = ChamferDistance()
        self.alpha = 0.99

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = PVNModel(
            embed_dim=point_cloud_model_embed_dim,  # 64
            in_channels=self.in_channels,  # 1027
            out_channels=self.out_channels,  # 3
        )
        # self.point_cloud_model = Tiger_Transformer_custom(num_classes=3, embed_dim=64, use_att=True,
        #                     dropout=0.1, extra_feature_channels=1024)


    def compute_eeg_loss(self, pc, cond, labels, shape_c=None):
        '''
        pc [:, :, :3] location info
        shape_c  is None
        '''
        x_0 = pc
        # print("x_0 shape", x_0.shape)  # ([4, 8192, 3])

        B, N, D = x_0.shape

        if shape_c is not None:
            print("---shape_c is not None---")
            noise_std = 0.02
            if np.random.random() > 0.5:
                shape_c = shape_c + torch.randn_like(shape_c) * noise_std
            shape_condition = shape_c

        # Sample random noise
        noise = torch.randn_like(x_0)


        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        timestep_scalar = timestep[0].item()

        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep)
        # print("x_t shape", x_t.shape)  # ([4, 8192, 3])

        if shape_c is not None:
            x_t_input = [shape_condition, x_t]
        else:
            x_t_input = [x_t]

        noise_std2 = 0.004
        if np.random.random() > 0.5:
            cond = cond + torch.randn_like(cond) * noise_std2

        cond = cond.unsqueeze(1).expand(-1, N, -1)

        x_t_input.append(cond)

        if shape_c is not None:
            x_t_input.append(color_result_soft.unsqueeze(1).expand(-1, N, -1))

        x_t_input = torch.cat(x_t_input, dim=2)  # 加噪后的x_t, eeg_feat
        # print("x_t_input shape", x_t_input.shape)  # ([4, 8192, 1027])

        # Forward
        # noise_pred = self.point_cloud_model(x_t_input.transpose(1, 2), timestep)
        noise_pred = self.point_cloud_model(x_t_input, timestep)




        if noise_pred.shape[-1] == 6:
            noise_pred = noise_pred[..., 3:]

        loss = F.mse_loss(noise_pred, noise)
        total_loss = loss




        if self.use_gan:
            fake_pc = self.scheduler.step(noise_pred, timestep_scalar, x_t).prev_sample


            fake_pc_detached = fake_pc.detach()

            # === GAN Loss ===
            real_labels = torch.ones(B, 1, device=pc.device)
            fake_labels = torch.zeros(B, 1, device=pc.device)

            real_logits = self.discriminator(x_0)
            fake_logits = self.discriminator(fake_pc_detached)

            d_loss_real = self.gan_loss(real_logits, real_labels)
            d_loss_fake = self.gan_loss(fake_logits, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            # # === 生成器的 adversarial loss ===
            # g_adv_logits = self.discriminator(fake_pc)  # 不用 detach
            # g_adv_loss = self.gan_loss(g_adv_logits, real_labels)


            return total_loss, d_loss, fake_pc
        else:
            return loss, None, None

    def compute_val_loss(self, pc, cond):
        """
        Validation version: No GAN, only MSE loss between predicted noise and actual noise.
        """
        x_0 = pc
        B, N, D = x_0.shape

        noise = torch.randn_like(x_0)
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)


        x_t = self.scheduler.add_noise(x_0, noise, timestep)

        cond = cond.unsqueeze(1).expand(-1, N, -1)
        x_t_input = torch.cat([x_t, cond], dim=2)


        # noise_pred = self.point_cloud_model(x_t_input.transpose(1, 2), timestep)
        noise_pred = self.point_cloud_model(x_t_input, timestep)


        if noise_pred.shape[-1] == 6:
            noise_pred = noise_pred[..., 3:]


        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def generate_eeg(self, num_points, cond, shape_c=None, fea_list=None, labels=None,
                     scheduler: Optional[str] = 'ddpm',
                     num_inference_steps: Optional[int] = 1000,
                     eta: Optional[float] = 0.0,
                     return_sample_every_n_steps: int = -1,
                     disable_tqdm: bool = False,
                     ):
        ## shape_c is None
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        N = num_points  # 8192

        # print("N",N)

        # B = eeg_data.shape[0]  # N
        B = cond.shape[0]  # N

        D = 3

        device = cond.device

        # Sample noise
        x_t = torch.randn(B, N, D, device=device)  # 随机高斯噪声

        # Set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())

        extra_set_kwargs = {"offset": 1} if accepts_offset else {}

        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())

        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # import pdb;pdb.set_trace()
        # eeg_video_fea, c1 = self.meta_eeg_video(eeg_data, 0)
        # eeg_point_fea, c2 = self.meta_eeg_point(eeg_data, 0)

        # video_acc_count, total, acc_list = self.retrieval_test(eeg_data, eeg_data2, fea_list, labels)
        #
        # eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(eeg_data, eeg_data2)

        # color_result_soft = color_result.softmax(1)

        if shape_c is not None:
            c1 = eeg_features2
        else:
            c1 = cond

        all_outputs = [x_t]

        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)

        for i, t in enumerate(progress_bar):
            if shape_c is not None:
                x_t_input = [shape_c, x_t]
            else:
                x_t_input = [x_t]
            # import pdb;pdb.set_trace()
            # c1_features = self.video_clip_mlp(c1).unsqueeze(1).expand(-1, N, -1)

            c1_features = c1.unsqueeze(1).expand(-1, N, -1)

            x_t_input.append(c1_features)

            if shape_c is not None:
                x_t_input.append(color_result_soft.unsqueeze(1).expand(-1, N, -1))

            x_t_input = torch.cat(x_t_input, dim=2)

            # Forward
            noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))


            # print("noise_pred shape", noise_pred.shape)

            # if noise_pred.shape[-1] == 6:
            #     noise_pred = noise_pred[..., 3:]

            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample


            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)

        output = x_t
        # return output, (video_acc_count, total, acc_list)
        return output

    def forward(self, pc, cond, labels, shape_c=None, mode: str = 'train', **kwargs):
        if mode == 'train':
            return self.compute_eeg_loss(pc, cond, labels, shape_c=None)

        elif mode == 'sample':
            return self.generate_eeg(2048, cond, shape_c)

        # elif mode == 'test_retrieval':
        #     return self.retrieval_test(eeg_data, eeg_data2, fea_list, labels)

        else:
            raise NotImplementedError()

