import torch
import torch.nn as nn
import numpy as np
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
POINTCNN_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, POINTCNN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
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
from PointCNN.model_PCNN.vae_gaussian_GAN import *
import math
import os
import datetime
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

args = parse_args()

if not args.checkpoint_path:
    raise ValueError("Please provide --checkpoint_path for inference.")

ckpt = torch.load(args.checkpoint_path, map_location=args.device)



# model = GaussianVAE(args).to(args.device)
model = GaussianVAE_GAN(args).to(args.device)

full_sd = model.state_dict()
for k, v in ckpt['model'].items():
    full_sd[k] = v

model.load_state_dict(full_sd, strict=True)

# model.load_state_dict(ckpt['model'])

test_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=False, point_path=args.ply_point_path)

# train_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=True, aug_data=True)

dataloader_test = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
# dataloader_train = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

save_dir = os.path.join(args.infer_output_dir, args.sub, args.experiment_name)
os.makedirs(save_dir, exist_ok=True)

with tqdm(dataloader_test, desc="Generating Point Clouds") as pbar:
    model.eval()
    for num_index in range(0, args.infer_repeat):
        time_b = time.time()
        video_acc_count_all, total_all = 0, 0
        for batch_idx, batch in enumerate(dataloader_test):

            point_c = None
            pc = batch['point_cloud'].to(args.device).float()[:, :, :3]
            eeg_data = batch['eeg_data'].to(args.device).float()
            eeg_data2 = batch['eeg_data2'].to(args.device).float()



            output = model.test_sample(args.num_points, eeg_data, eeg_data2)
            # batch_pcs = []

            for ii in range(0, output.shape[0]):
                # point_pred, point_lbl = output[ii].detach().cpu().numpy(), pc[ii].detach().cpu().numpy()
                point_pred, point_lbl = output[ii].detach().cpu().numpy(), pc[ii].detach().cpu().numpy()
                # batch_pcs.append(point_pred)
                pcd = o3d.geometry.PointCloud()


                pcd.points = o3d.utility.Vector3dVector(point_pred)

                name = batch['name'][ii]

                o3d.io.write_point_cloud(os.path.join(save_dir, f'{name}-{num_index}.ply'), pcd)



        time_e = time.time()
        print(time_e - time_b)


