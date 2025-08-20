import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn
from .common import *
import sys
import os

from eeg_data_process.extract_eeg_feature_double import VideoImageEEGClassifyColor3

from torch.utils.tensorboard import SummaryWriter

from PointGeneration.EEGTo3DDiffusionModel_simple import EEGTo3DDiffusionModel

from cls.pointnet2_cls_msg import get_model, get_loss

class Mind2Cloud(Module):

    def __init__(self, args):
        super().__init__()

        self.args = args




        self.diffusion = EEGTo3DDiffusionModel(beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule, sub=args.sub,
                                               point_cloud_model_embed_dim=args.point_cloud_model_embed_dim, in_channels=args.in_channels, out_channels=args.out_channels)


        self.eeg_encoder = VideoImageEEGClassifyColor3(num_channels=64, sequence_length=600, sequence_length2=250, num_latents=1024,cls_num=72)


    def get_loss(self, x, eeg_data1, eeg_data2, img_features, labels):

        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        mse_loss_fn = nn.MSELoss()

        eeg_feature = self.eeg_encoder(eeg_data1, eeg_data2)


        logit_scale = self.eeg_encoder.logit_scale

        loss_img = self.eeg_encoder.loss_func(eeg_feature, img_features, logit_scale)

        loss_regress = mse_loss_fn(eeg_feature, img_features)




        loss_eeg = loss_regress + loss_img

        cond = eeg_feature


        loss_dm, d_loss, fake_pc = self.diffusion(x, cond, labels, mode='train')


        alpha = 0.95
        loss = alpha * loss_dm * 10 + (1 - alpha) * loss_eeg * 10

        # loss = loss_dm + loss_eeg



        # return loss,  loss_dm, loss_eeg
        return loss, loss_dm, loss_eeg, d_loss, fake_pc

    # def test_loss(self, x, eeg_data1, eeg_data2, img_features, text_features):


    def get_val_loss(self, x, eeg_data1, eeg_data2, img_features):

        """
        Validation-time loss function without adversarial/GAN losses.
        Returns:
            loss: total combined loss
            loss_dm: diffusion model (reconstruction) loss
            loss_pc: eeg-image alignment loss
        """
        mse_loss_fn = nn.MSELoss()

        with torch.no_grad():
            # EEG feature encoding
            eeg_feature = self.eeg_encoder(eeg_data1, eeg_data2)
            # eeg_feature = self.eeg_encoder(eeg_data1)

            logit_scale = self.eeg_encoder.logit_scale

            # EEG alignment loss with image feature
            loss_img = self.eeg_encoder.loss_func(eeg_feature, img_features, logit_scale)
            loss_regress = mse_loss_fn(eeg_feature, img_features)

            # points = x.transpose(2, 1)
            # _, trans_feat = self.classifier(points)

            # trans_feat = trans_feat.squeeze(-1)
            # loss_attn = mse_loss_fn(eeg_feature, trans_feat)

            # loss_eeg = 0.5 * loss_img + 0.25 * loss_regress + 0.25 * loss_attn

            # loss_eeg = (1 - alpha) * loss_img * 10 + alpha * loss_regress * 10
            loss_eeg = loss_img + loss_regress

            # Generate EEG condition
            cond = eeg_feature

            # Diffusion loss only (no GAN)
            loss_dm = self.diffusion.compute_val_loss(x, cond)

            alpha = 0.95
            total_loss = alpha * loss_dm * 10 + (1 - alpha) * loss_eeg * 10
            # total_loss = loss_dm + loss_eeg

        return total_loss, loss_dm, loss_eeg




    def test_sample(self, num_points, eeg_data1, eeg_data2):
    # def test_sample(self, num_points, eeg_data1):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        eeg_feature = self.eeg_encoder(eeg_data1, eeg_data2)
        # eeg_feature = self.eeg_encoder(eeg_data1)

        sample = self.diffusion(num_points, eeg_feature, None, mode='sample')

        # print("num_points", num_points)

        return sample
