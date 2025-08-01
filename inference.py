import torch
import torch.nn as nn
import numpy as np
import os, sys
from pathlib import Path
from contextlib import nullcontext
from accelerate import Accelerator
from train_utils.parse import parse_args
from train_utils import training_utils
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import datetime
from eeg_data_process.EEGdataset import AllDataFeatureTwoEEG
from eeg_data_process.clip_loss import ClipLoss
import open3d as o3d
from accelerate import DistributedDataParallelKwargs
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PointCNN.model_PCNN.vae_gaussian import *
import math
import os
import datetime
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

args = parse_args()


ckpt = torch.load(" ",  map_location=args.device)

model = Mind2Cloud(args).to(args.device)



model.load_state_dict(ckpt['model'])



test_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=False, point_path=args.ply_point_path)



dataloader_test = DataLoader(test_set, batch_size=48, shuffle=False, num_workers=args.num_workers)


save_dir = ""
os.makedirs(save_dir, exist_ok=True)


with tqdm(dataloader_test, desc="Generating Point Clouds") as pbar:
    model.eval()
    for num_index in range(0, 5):
        time_b = time.time()
        video_acc_count_all, total_all = 0, 0
        for batch_idx, batch in enumerate(dataloader_test):

            point_c = None
            pc = batch['point_cloud'].to(args.device).float()[:, :, :3]
            eeg_data = batch['eeg_data'].to(args.device).float()
            eeg_data2 = batch['eeg_data2'].to(args.device).float()



            output = model.test_sample(2048, eeg_data, eeg_data2)
            # batch_pcs = []

            for ii in range(0, output.shape[0]):
                # point_pred, point_lbl = output[ii].detach().cpu().numpy(), pc[ii].detach().cpu().numpy()
                point_pred, point_lbl = output[ii].detach().cpu().numpy(), pc[ii].detach().cpu().numpy()
                # batch_pcs.append(point_pred)
                pcd = o3d.geometry.PointCloud()


                pcd.points = o3d.utility.Vector3dVector(point_pred)

                name = batch['name'][ii]

                o3d.io.write_point_cloud(f'{save_dir}/{name}-{num_index}.ply', pcd)



        time_e = time.time()
        print(time_e - time_b)


