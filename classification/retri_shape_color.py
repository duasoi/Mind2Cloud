import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange
import random
import wandb
import sys
import argparse
sys.path.append('.')
# from eeg_data_process.extract_eeg_feature_itrans import VideoImageEEGClassifyColor3
# from eeg_data_process.extract_eeg_feature_mamba import VideoImageEEGClassifyColor3
# from eeg_data_process.extract_eeg_feature import VideoImageEEGClassifyColor3
from eeg_data_process.extract_eeg_feature_merge import VideoImageEEGClassifyColor3

from eeg_data_process.EEGdataset import AllDataFeatureTwoEEG
from eeg_data_process.clip_loss import ClipLoss


###python classification/retri_shape_color.py --model 'VideoImageEEGClassifyColor3'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class_weights = torch.tensor([1.0, 1.0, 0.1, 0.1, 1.0, 1.0])


def get_result(labels, cls_result, K=5):
    cls_result_soft = cls_result.softmax(1)
    _, predicted = torch.max(cls_result_soft, 1)
    correct = (predicted == labels).sum().item()
    
    topk_pred = cls_result_soft.topk(K, dim=1)[1]
    top_5_correct = topk_pred.eq(labels.view(-1, 1).expand_as(topk_pred)).sum().item()
    coarse_num = top_5_correct
    return correct, top_5_correct, coarse_num

feature_cls = 'color_video_fea' # color_video_fea, color_point_fea, gray_video_fea, gray_point_fea, txt_fea

def train_model(config, eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, lr_scheduler):

    eeg_model.train()
    text_features_all = text_features_all[:, 0].to(device).float()
    # print("text_features_all", text_features_all.shape) # (72,1024)
    img_features_all = img_features_all[:, 0].to(device).float()
    # print("img_features_all", img_features_all.shape)  # (72,1024)
    total_loss = 0
    
    shape_correct, shape_top5_correct = 0, 0
    color_correct, color_top2_correct = 0, 0
    retri_correct, retri_top5_correct = 0, 0

    total = 0

    alpha = 0.95
    mse_loss_fn = nn.MSELoss()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss(class_weights.to(device))

    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx % 50 == 0:
            print(batch_idx)
        
        eeg_data = batch_data['eeg_data'].to(device).float()
        # print(f"eeg data shape is {eeg_data.shape}") # ([128, 64, 600])
        eeg_data2 = batch_data['eeg_data2'].to(device).float()
        # print(f"eeg data 2 shape is {eeg_data2.shape}") # ([128, 64, 250])


        img_features = batch_data[feature_cls].to(device).float()
        # print("img_features shape", img_features.shape)  # ([128, 1024])

        text_features = batch_data['txt_fea'].to(device).float()
        # print("text_features shape", text_features.shape)  # ([128, 1024])


        labels_color = batch_data['color_label'].to(device).long()
        # print("labels_color", labels_color.shape) # ([128]) 颜色标签

        labels_shape = batch_data['cls_label'].to(device)
        # print("labels_shape", labels_shape.shape)  # # ([128]) 种类标签
        
        optimizer.zero_grad()

        _, eeg_features1, shape_result, eeg_features2, color_result = eeg_model(eeg_data, eeg_data2)

                
        # features_list.append(eeg_features)
        #eeg_model.loss_func(eeg_features2, img_features, logit_scale) + 
        logit_scale = eeg_model.logit_scale

        # img_loss = eeg_model.loss_func(eeg_features1, img_features, logit_scale) + eeg_model.loss_func(eeg_features2, img_features, logit_scale)
        img_loss = eeg_model.loss_func(eeg_features1, img_features) + eeg_model.loss_func(eeg_features2, img_features)
        text_loss = mse_loss_fn(eeg_features1, text_features) + mse_loss_fn(eeg_features2, text_features)

        contrastive_loss = img_loss

        regress_loss = mse_loss_fn(eeg_features2, img_features) + mse_loss_fn(eeg_features1, img_features)

        # print("color_result, labels_color, shape_result, labels_shape", color_result.shape, labels_color.shape, shape_result.shape, labels_shape.shape)
        #         ([128, 6])     ([128])        ([128, 72])    ([128])

        loss_cls = criterion2(color_result, labels_color) + criterion1(shape_result, labels_shape)

        loss = alpha * regress_loss * 10 + (1 - alpha) * contrastive_loss * 10 + loss_cls * 0.5 + text_loss
        # loss = loss_cls * 0.1
        loss.backward()
        optimizer.step()
        total += labels_shape.shape[0]

        # 预测类别[128, 72], 每个样本(共128)在 72 个类别中的最大预测值索引
        logits_img = logit_scale * eeg_features1 @ img_features_all.T   # ([128, 1024])  ([72, 1024])

        logits_single = logits_img

        predicted = torch.argmax(logits_single, dim=1)  # (n_batch, ) \in {0, 1, ..., n_cls-1}

        retri_correct += (predicted == labels_shape).sum().item()

        total_loss += loss.item()

        # 计算分类准确率（Top-1 和 Top-5）
        co1, co2, co3 = get_result(labels_shape, shape_result, 5)   # ([N])    ([N, 72])
        shape_correct += co1
        shape_top5_correct += co2

        # 计算颜色分类的 Top-1 和 Top-2
        co1, co2, co3 = get_result(labels_color, color_result, 2)  # [N], [N,6]
        color_correct += co1
        color_top2_correct += co2
        
        lr_scheduler.step()

    average_loss = total_loss / (batch_idx+1)

    shape_acc, shape_top5_acc = shape_correct / total, shape_top5_correct / total

    color_acc, color_top2_acc = color_correct / total, color_top2_correct / total

    retri_acc, retri_top5_acc = retri_correct / total, retri_top5_correct / total

    return average_loss, retri_acc, retri_top5_acc, shape_acc, shape_top5_acc, color_acc, color_top2_acc

def evaluate_model(config, eeg_model, dataloader, device, text_features_all, img_features_all):
    eeg_model.eval()
    
    if len(img_features_all.shape) == 3:
        img_features_all = img_features_all[:, 0].to(device).float()
        text_features_all = text_features_all[:, 0].to(device).float()
    else:
        img_features_all = img_features_all.to(device).float()
        text_features_all = text_features_all.to(device).float()
    
    total_loss = 0
    total = 0
    shape_correct, shape_top5_correct = 0, 0
    color_correct, color_top2_correct = 0, 0
    retri_correct, retri_top5_correct = 0, 0
    
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss(class_weights.to(device))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            eeg_data = batch_data['eeg_data'].to(device).float()
            eeg_data2 = batch_data['eeg_data2'].to(device).float()

            img_features = batch_data[feature_cls].to(device).float()
            # img_features = batch_data['gray_point_fea'].to(device).float()

            labels_color = batch_data['color_label'].to(device).long()  # [N]

            labels_shape = batch_data['cls_label'].to(device)  # # [N]
            

            _, eeg_features1, shape_result, eeg_features2, color_result = eeg_model(eeg_data, eeg_data2)

            loss_cls = criterion2(color_result, labels_color) + criterion1(shape_result, labels_shape)

            logit_scale = eeg_model.logit_scale 

            loss = loss_cls * 0.1
            total_loss += loss.item()
            total += labels_shape.shape[0]
            
            co1, co2, co3 = get_result(labels_shape, shape_result, 5)
            shape_correct += co1
            shape_top5_correct += co2

            co1, co2, co3 = get_result(labels_color, color_result, 2)
            color_correct += co1
            color_top2_correct += co2
            
            for idx, label in enumerate(labels_shape):

                logits_text = logit_scale * eeg_features1[idx] @ img_features_all.T  
                logits_single = logits_text
                predicted_label = torch.argmax(logits_single).item() # (n_batch, ) \in {0, 1, ..., n_cls-1}

                if predicted_label == label.item():
                    retri_correct += 1
                _, top5_indices = torch.topk(logits_single, 5, largest =True)                             
                if label.item() in [i for i in top5_indices.tolist()]:                
                    retri_top5_correct += 1
            
    average_loss = total_loss / (batch_idx+1)

    shape_acc, shape_top5_acc = shape_correct / total, shape_top5_correct / total
    color_acc, color_top2_acc = color_correct / total, color_top2_correct / total
    retri_acc, retri_top5_acc = retri_correct / total, retri_top5_correct / total

    return average_loss, retri_acc, retri_top5_acc, shape_acc, shape_top5_acc, color_acc, color_top2_acc

def adjust_lr(optimizer, epoch, lr, config):
    lr_c = lr * ((1 - epoch/(config['epochs'] + 10)) ** 0.9)
    print("lr", lr_c)
    for p in optimizer.param_groups:
        p['lr'] = lr_c

def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config):

    save_model_path = f"{config['save_path']}/{sub}/cls_model{current_time}"
    os.makedirs(save_model_path, exist_ok=True)

    # record_log = open(config['save_path'] + '/result_color_shape_all.txt{current_time}', 'a')
    # record_log = open(f"{config['save_path']}/{sub}/result_color_shape_all_{current_time}.txt", 'a')
    record_log = open(f"{config['save_path']}/result_color_shape_all_{current_time}.txt", 'a')

    log_f = open(save_model_path + '/log.txt', 'w')
    for key in config.keys():
        log_f.write(f"{key}:{config[key]}\n")
    log_f.flush()
    
    train_losses, test_losses = [], []
    train_retri_accuracies, train_shape_accuracies, train_color_accuracies = [], [], []
    test_retri_accuracies, test_shape_accuracies, test_color_accuracies = [], [], []
    
    best_shape_acc, best_shape_top5_acc, best_shape_top5_acc_all = 0.0, 0.0, 0.0
    best_color_acc, best_color_top2_acc, best_color_top2_acc_all = 0.0, 0.0, 0.0
    best_retri_acc, best_retri_top5_acc, best_retri_top5_acc_all = 0.0, 0.0, 0.0

    results = []  
    total_steps = int((config['epochs']+5) * len(train_dataloader))
    # import pdb;pdb.set_trace()
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        total_steps=total_steps,
        final_div_factor=10000,
        last_epoch=-1, pct_start=2 / config['epochs']
    )
    for epoch in range(config['epochs']):  # 200
        # if epoch == 0:
        #     test_loss, \
        #         test_retri_acc, \
        #         test_retri_top5_acc, \
        #         test_shape_acc, \
        #         test_shape_top5_acc, \
        #         test_color_acc, \
        #         test_color_top2_acc = evaluate_model(
        #         config, eeg_model, test_dataloader, device,
        #         text_features_test_all, img_features_test_all
        #     )
        #     print("test val overfit :", test_loss, test_retri_acc, test_shape_acc, test_color_acc)
        #
        #     trian_loss, \
        #         train_retri_acc, \
        #         train_retri_top5_acc, \
        #         train_shape_acc, \
        #         train_shape_top5_acc, \
        #         train_color_acc, \
        #         train_color_top2_acc = evaluate_model(
        #         config, eeg_model, train_dataloader, device,
        #         text_features_train_all, img_features_train_all
        #     )
        #     print("train val overfit :", trian_loss, train_retri_acc, train_shape_acc, train_color_acc)

        train_loss, \
            train_retri_acc, \
            train_retri_top5_acc, \
            train_shape_acc, \
            train_shape_top5_acc, \
            train_color_acc, \
            train_color_top2_acc = train_model(
            config, eeg_model, train_dataloader, optimizer, device,
            text_features_train_all, img_features_train_all, lr_scheduler
        )

        if (epoch + 1) % 20 == 0:
            file_path = f"{save_model_path}/{epoch + 1}.pth"
            torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")

        test_loss, \
            test_retri_acc, \
            test_retri_top5_acc, \
            test_shape_acc, \
            test_shape_top5_acc, \
            test_color_acc, \
            test_color_top2_acc = evaluate_model(
            config, eeg_model, test_dataloader, device,
            text_features_test_all, img_features_test_all
        )

        train_losses.append(train_loss)
        train_retri_accuracies.append(train_retri_acc)
        train_shape_accuracies.append(train_shape_acc)
        train_color_accuracies.append(train_color_acc)

        test_losses.append(test_loss)
        test_retri_accuracies.append(test_retri_acc)
        test_shape_accuracies.append(test_shape_acc)
        test_color_accuracies.append(test_color_acc)
        
        # Append results for this epoch
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_retri_acc": train_retri_acc,
            "train_shape_acc": train_shape_acc,
            "train_color_acc": train_color_acc,
            "test_loss": test_loss,
            "test_retri_acc": test_retri_acc,
            "test_shape_acc": test_shape_acc,
            "test_color_acc": test_color_acc,
        }

        results.append(epoch_results)
        
        if test_retri_acc > best_retri_acc or (test_retri_acc == best_retri_acc and test_retri_top5_acc > best_retri_top5_acc):
            best_retri_top5_acc = test_retri_top5_acc
            best_retri_acc = test_retri_acc
            file_path = f"{save_model_path}/best-retri.pth"
            torch.save(eeg_model.state_dict(), file_path)            
            print(f"model saved in {file_path}!")
        
        if test_shape_acc > best_shape_acc or (test_shape_acc == best_shape_acc and test_shape_top5_acc > best_shape_top5_acc):
            best_shape_acc = test_shape_acc
            best_shape_top5_acc = test_shape_top5_acc
            file_path = f"{save_model_path}/best-shape.pth"
            torch.save(eeg_model.state_dict(), file_path)            
            print(f"model saved in {file_path}!")
        
        if test_color_acc > best_color_acc or (test_color_acc == best_color_acc and test_color_top2_acc > best_color_top2_acc):
            best_color_acc = test_color_acc
            best_color_top2_acc = test_color_top2_acc
            file_path = f"{save_model_path}/best-color.pth"
            torch.save(eeg_model.state_dict(), file_path)            
            print(f"model saved in {file_path}!")
        
        
        best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_retri_acc": train_retri_acc,
                "train_shape_acc": train_shape_acc,
                "train_color_acc": train_color_acc,
                "test_loss": test_loss,
                "test_retri_acc": best_retri_acc,
                "test_shape_acc": best_shape_acc,
                "test_color_acc": best_color_acc,
            }
        best_retri_top5_acc_all = max([best_retri_top5_acc_all, test_retri_top5_acc])
        best_shape_top5_acc_all = max([best_shape_top5_acc_all, test_shape_top5_acc])
        best_color_top2_acc_all = max([best_color_top2_acc_all, test_color_top2_acc])

        ss = f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n"
        ss += f"train_retri_acc: {train_retri_acc:.4f}, train_shape_acc: {train_shape_acc:.4f}, train_color_acc: {train_color_acc:.4f}\n"
        ss += f"test_retri_acc: {test_retri_acc:.4f}. test_shape_acc: {test_shape_acc:.4f}, test_color_acc: {test_color_acc:.4f}\n"
        ss += f"best_retri_acc: {best_retri_acc:.4f}, best_shape_acc: {best_shape_acc:.4f}, best_color_acc: {best_color_acc:.4f}\n"
        ss += f"test_retri_top5_acc: {test_retri_top5_acc:.4f}, test_shape_top5_acc: {test_shape_top5_acc:.4f}, test_color_top2_acc: {test_color_top2_acc:.4f}\n"
        ss += f"best_retri_top5_acc_all: {best_retri_top5_acc_all:.4f}, best_shape_top5_acc_all: {best_shape_top5_acc_all:.4f}, best_color_top2_acc_all: {best_color_top2_acc_all:.4f}\n"
        print(ss)
        log_f.write(ss)
        log_f.flush()

    log_f.close()
    ss = f"{sub}, retri_color_shape_{current_time}_{config['model']}_{feature_cls}_time_len{config['time']}:\n"
    ss += f"best_retri_acc: {best_retri_acc:.4f}, best_shape_acc: {best_shape_acc:.4f}, best_color_acc: {best_color_acc:.4f}\n"
    ss += f"test_retri_top5_acc: {test_retri_top5_acc:.4f}, best_shape_top5_acc: {best_shape_top5_acc:.4f}, test_color_top2_acc: {test_color_top2_acc:.4f}\n"
    ss += f"best_retri_top5_acc_all: {best_retri_top5_acc_all:.4f}, best_shape_top5_acc_all: {best_shape_top5_acc_all:.4f}, best_color_top2_acc_all: {best_color_top2_acc_all:.4f}\n"
    record_log.write(ss + '\n')
    record_log.flush()
    record_log.close()
    return results

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="/data1/hxf/neuro-3D-main/EEG_3Datasets/")
    parser.add_argument('--model', type=str, default="VideoImageEEGClassifyColor3")
    parser.add_argument('--time_cls', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=600)
    parser.add_argument('--sub', type=str, default='sub13')
    opt = parser.parse_args()
    return opt

def main(args, sub='sub03'):
    set_seed(0)
    config = {
        "save_path": args.root_path + "model/retraival/",
        "data_path": args.root_path,
        "lr": 1e-3,
        "epochs": 200,
        "batch_size": 128,
        "time_len1": 600,
        "time_len2": 250,
        "model": args.model,
        "time": args.time_cls
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_latents = 1024

    if config["model"] == "VideoImageEEGClassifyColor3":
        eeg_model = VideoImageEEGClassifyColor3(num_channels=64, sequence_length=config['time_len1'], sequence_length2=config['time_len2'], num_latents=num_latents, cls_num=72)
    eeg_model.to(device)
    
    print(get_parameter_number(eeg_model))
    optimizer = torch.optim.AdamW(eeg_model.parameters(), lr=config['lr'], weight_decay=5e-4)#, weight_decay=5e-4
    # optimizer = torch.optim.Adam(eeg_model.parameters(), lr=config['lr'])#, weight_decay=5e-4

    train_dataset = AllDataFeatureTwoEEG(config['data_path'], sub_list=[sub], train=True, aug_data=True)
    # print(f"train total: {len(train_dataset)}")  # 1152 = 72 * 8 * 2

    # train_dataset = AllDataFeatureTwoEEG(config['data_path'], sub_list=['sub09', 'sub10', 'sub11', 'sub12', 'sub13', 'sub14'], train=True, aug_data=True)
    test_dataset = AllDataFeatureTwoEEG(config['data_path'], sub_list=[sub], train=False)
    # print(f"train len:{train_dataset.__len__()}, test len:{test_dataset.__len__()}")  # 1152  144
    
    txt_features_train_all = train_dataset.txt_features  # ([72, 8, 1024])

    txt_features_test_all = test_dataset.txt_features  # ([72, 2, 1024])

    if feature_cls == 'color_video_fea':
        img_features_train_all = train_dataset.color_video_features  # clip
        # print("img_features_train_all", img_features_train_all.shape)  # ([72, 8, 1024])
        img_features_test_all = test_dataset.color_video_features
        # print("img_features_test_all", img_features_test_all.shape)  # ([72, 2, 1024])

    elif feature_cls == 'color_point_fea':
        img_features_train_all = train_dataset.color_point_features
        img_features_test_all = test_dataset.color_point_features
    elif feature_cls == 'gray_video_fea':
        img_features_train_all = train_dataset.gray_video_features
        img_features_test_all = test_dataset.gray_video_features
    elif feature_cls == 'gray_point_fea':
        img_features_train_all = train_dataset.gray_point_features
        img_features_test_all = test_dataset.gray_point_features
    elif feature_cls == 'txt_fea':
        img_features_train_all = train_dataset.txt_features
        img_features_test_all = test_dataset.txt_features
    # import pdb;pdb.set_trace()
    
    # print(img_features_train_all.shape, img_features_test_all.shape, txt_features_train_all.shape, txt_features_test_all.shape)
        # ([72, 8, 1024])                ([72, 2, 1024])                  ([72, 8, 1024])               ([72, 2, 1024])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)


    test_loader = DataLoader(test_dataset, batch_size=72, shuffle=False, num_workers=0, drop_last=True)

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M"
                                                    )
    results = main_train_loop(sub, current_time, eeg_model, train_loader, test_loader, optimizer, device, 
                                  txt_features_train_all, txt_features_test_all, img_features_train_all, img_features_test_all, config)

if __name__ == '__main__':
    args = parse_args()
    main(args, args.sub)
