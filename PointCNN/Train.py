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
from train_utils import training_utils
from train_utils.parse import parse_args
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import datetime
from eeg_data_process.EEGdataset import AllDataFeatureTwoEEG
from eeg_data_process.clip_loss import ClipLoss
import open3d as o3d
from accelerate import DistributedDataParallelKwargs
import time

from model_PCNN.metrics import *
from model_PCNN.metrics import _pairwise_EMD_CD_, _jsdiv
from PointCNN.model_PCNN.vae_gaussian_GAN import *

# from PointCNN.model_PCNN.vae_flow import *
# from PointCNN.model_PCNN.flow import add_spectral_norm, spectral_norm_power_iteration

###  debug V100
import os
import datetime
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main():
    args = parse_args()
    training_utils.set_seed(args.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=False,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              kwargs_handlers=[ddp_kwargs])

    training_utils.setup_distributed_print(accelerator.is_main_process)

    model = GaussianVAE_GAN(args).to(args.device)
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')

    # optimizer = training_utils.get_optimizer(args, model)
    generator_optimizer = training_utils.get_optimizer(args, model)
    generator_scheduler = training_utils.get_scheduler(args, generator_optimizer)

    discriminator_optimizer = training_utils.get_optimizer(
        args, model.diffusion.discriminator
    )
    discriminator_scheduler = training_utils.get_scheduler(args, discriminator_optimizer)

    train_state = training_utils.resume_from_checkpoint(args, model, generator_optimizer, generator_scheduler)
    print("subject", args.sub)

    train_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=True, aug_data=True)
    test_set = AllDataFeatureTwoEEG(args.data_path, sub_list=[args.sub], train=False, point_path=args.ply_point_path)

    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_test = DataLoader(test_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)

    model_save_path = os.path.join(args.output_dir, args.sub, args.experiment_name)
    os.makedirs(model_save_path, exist_ok=True)

    log_info_txt = open(os.path.join(model_save_path, 'log.txt'), 'w')
    writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'tensorboard_logs'))
    print(f'Model save path: {model_save_path}')

    best_val_loss = float('inf')
    total_start_time = time.time()

    while train_state.step < args.max_steps:
        for batch in dataloader_train:
            model.train()

            pc = batch['point_cloud'].to(args.device).float()[:, :, :3]
            # print("pc shape", pc.shape)

            eeg_data = batch['eeg_data'].to(args.device).float()

            labels = batch['cls_label'].to(args.device)
            # print("eeg_data shape", eeg_data.shape)

            eeg_data2 = batch['eeg_data2'].to(args.device).float()
            img_features = batch['color_video_fea'].to(args.device).float()
            labels_shape = batch['cls_label'].to(args.device)
            # text_features = batch['txt_fea'].to(args.device).float()
            # point_features = batch['color_point_fea'].to(args.device).float()

            with accelerator.accumulate(model):
                # loss, loss_dm, loss_pc, d_loss, fake_pc = model.get_loss(pc, eeg_data, eeg_data2, img_features, labels_shape)
                loss, loss_dm, loss_pc, d_loss, fake_pc = model.get_loss(pc, eeg_data, eeg_data2, img_features, labels)


                if d_loss is not None:

                    discriminator_optimizer.zero_grad()
                    accelerator.backward(d_loss)
                    if accelerator.sync_gradients:
                        discriminator_optimizer.step()
                        discriminator_scheduler.step()

                    # === G adversarial loss (推迟判别器前向传播) ===
                g_adv_loss = None
                if fake_pc is not None:
                    real_labels = torch.ones(pc.size(0), 1, device=pc.device)
                    g_adv_logits = model.diffusion.discriminator(fake_pc)
                    g_adv_loss = model.diffusion.gan_loss(g_adv_logits, real_labels)
                    loss = loss + 0.1 * g_adv_loss  # 加上 adversarial loss

                # === 生成器优化 ===
                generator_optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    generator_optimizer.step()
                    generator_scheduler.step()
                    train_state.step += 1



            # Logging and validation
            if accelerator.sync_gradients and accelerator.is_main_process:
                if train_state.step % args.log_step_freq == 0:
                    val_loss, val_loss_dm, val_loss_pc = retrieval_test(
                        args, model, dataloader_test)

                    best_val_loss = min(best_val_loss, val_loss)

                    writer.add_scalar('Loss/Train_Total', loss.item(), train_state.step)
                    writer.add_scalar('Loss/Train_DM', loss_dm.item(), train_state.step)
                    writer.add_scalar('Loss/Train_PC', loss_pc.item(), train_state.step)
                    # writer.add_scalar('Loss/Train_CD', loss_cd.item(), train_state.step)
                    writer.add_scalar('Loss/Train_G', g_adv_loss.item(), train_state.step)
                    writer.add_scalar('Loss/Train_D', d_loss.item(), train_state.step)

                    writer.add_scalar('Learning_Rate', generator_optimizer.param_groups[0]['lr'], train_state.step)

                    writer.add_scalar('Loss/Val_Total', val_loss, train_state.step)
                    writer.add_scalar('Loss/Val_DM', val_loss_dm, train_state.step)
                    writer.add_scalar('Loss/Val_PC', val_loss_pc, train_state.step)
                    # writer.add_scalar('Loss/Val_CD', val_loss_cd, train_state.step)
                    # writer.add_scalar('Loss/Val_G', val_g_loss, train_state.step)
                    # writer.add_scalar('Loss/Val_D', val_d_loss, train_state.step)

                    log_msg = (
                        f"\n[Step {train_state.step:07d}] "
                        f"LR: {generator_optimizer.param_groups[0]['lr']:.6f}, "
                        f"Train Loss: {loss.item():.4f}, DM Loss: {loss_dm.item():.4f}, PC Loss: {loss_pc.item():.4f}, g_loss: {g_adv_loss.item(): .4f}, d_loss: {d_loss.item(): .4f}\n"
                        f"Validation Loss: {val_loss:.4f}, val_DM Loss: {val_loss_dm:.4f}, val_PC Loss: {val_loss_pc:.4f}, Best Val Loss: {best_val_loss:.4f}\n"
                    )
                    print(log_msg)
                    log_info_txt.write(log_msg)
                    log_info_txt.flush()

                if train_state.step % args.checkpoint_freq == 0 or train_state.step == 1:
                    checkpoint = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'generator_optimizer': generator_optimizer.state_dict(),
                        'generator_scheduler': generator_scheduler.state_dict(),
                        'epoch': train_state.epoch,
                        'step': train_state.step,
                        'best_val': best_val_loss
                    }
                    checkpoint_path = os.path.join(model_save_path, f'checkpoint-{train_state.step}.pth')
                    accelerator.save(checkpoint, checkpoint_path)
                    print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')

            if train_state.step >= args.max_steps:
                break

    if accelerator.is_main_process:
        log_info_txt.write(f'\nTraining completed at {time.ctime()}\n')
        log_info_txt.flush()
        log_info_txt.close()
        writer.close()

    print(f"\nTraining completed in {time.time() - total_start_time:.2f} seconds.")


def retrieval_test(args, model, dataloader_test):
    model.eval()
    total_val_loss = total_loss_dm = total_loss_pc = total_loss_cd = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader_test:
            pc = batch['point_cloud'].to(args.device).float()[:, :, :3]


            eeg_data = batch['eeg_data'].to(args.device).float()
            eeg_data2 = batch['eeg_data2'].to(args.device).float()
            img_features = batch['color_video_fea'].to(args.device).float()
            labels_shape = batch['cls_label'].to(args.device)
            # text_features = batch['txt_fea'].to(args.device).float()
            # point_features = batch['color_point_fea'].to(args.device).float()

            # val_loss, loss_dm, loss_pc = model.get_loss(pc, eeg_data, eeg_data2, img_features)
            # val_loss, loss_dm, loss_pc = model.get_val_loss(pc, eeg_data, eeg_data2, img_features, labels_shape)
            val_loss, loss_dm, loss_pc = model.get_val_loss(pc, eeg_data, eeg_data2, img_features)
            total_val_loss += val_loss.item()
            total_loss_dm += loss_dm.item()
            total_loss_pc += loss_pc.item()
            # total_loss_cd += loss_cd.item()
            # total_g_adv_loss += g_adv_loss.item()
            # total_d_loss += d_loss.item()
            count += 1

    model.train()
    return total_val_loss / count, total_loss_dm / count, total_loss_pc / count


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()

