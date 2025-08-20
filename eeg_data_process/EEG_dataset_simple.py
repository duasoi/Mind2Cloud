from torch.utils.data import Dataset
import os
import numpy as np
import torch
import open3d as o3d
# import pandas as pd

class AllDataFeatureTwoEEG(Dataset):

    def __init__(self, data_path, sub_list, train=True, time_len=250, test_mean=True, aug_data=False, point_path=''):
        self.data_path = data_path
        self.sub_list = sub_list
        self.train = train
        self.time_len = time_len
        self.test_mean = test_mean
        self.aug_data = aug_data

        # ------划分视频训练与测试集------
        all_name_list = sorted(os.listdir(self.data_path + 'video_new/'))
        self.name_list = []

        remove_name = ['08', '09'] if train else ['00', '01', '02', '03', '04', '05', '06', '07']

        for name in all_name_list:
            if name.endswith('.mp4') and name[-6:-4] not in remove_name:
                self.name_list.append(name[:-4])

        self.name_list = np.array(self.name_list).reshape(72, -1)
        # print("-----", self.name_list.shape) # (72, 8) (72, 2)
        # ------划分视频训练与测试集------

        self.point_data = []
        # self.point_data = np.zeros((self.name_list.shape[0], self.name_list.shape[1], 8192, 6))
        # point_path = point_path or self.data_path + 'point_cloud_simple/'
        point_path = point_path or self.data_path + 'pc_2048/'

        for ii in range(self.name_list.shape[0]):
            row = []
            for jj in range(self.name_list.shape[1]):
                pc = np.load(point_path + self.name_list[ii][jj][3:] + '.npy')  # shape: (8192, 6)
                row.append(pc[np.newaxis, ...])  # shape: (1, 8192, 6)
            self.point_data.append(np.concatenate(row, axis=0))  # (num_obj, 8192, 6)

        self.point_data = np.stack(self.point_data, axis=0)  # shape: (num_cls, num_obj, 8192, 6)

        # print("self.point_data shape", self.point_data.shape)  # 包含训练集(72, 8, 8192, 6)和测试集(72, 2, 8192, 6)

        self.normalize_point_cloud()

        self.eeg_data, self.eeg_data2 = self.load_eeg()

        if not self.train:
            if self.test_mean:
                self.eeg_data = np.mean(self.eeg_data, axis=3, keepdims=True)
                self.eeg_data2 = np.mean(self.eeg_data2, axis=3, keepdims=True)
                self.obj_num, self.trails_num = 2, 1
            else:
                self.obj_num, self.trails_num = 2, 4
        else:
            self.obj_num, self.trails_num = 8, 2

        self.cls_num = 72
        self.clip_features = torch.load(self.data_path + 'clip_feature.pth')
        for key in self.clip_features.keys():
            self.clip_features[key]['point'] /= 8.0

    def normalize_point_cloud(self):
        """
        对 self.point_data 中的点云数据进行全局归一化处理：
        - xyz 坐标全局标准化为 0 均值，1 标准差
        - 其他特征归一化到 [-1, 1]
        """
        B, O, N, C = self.point_data.shape  # (类别数, 每类样本数, 点数, 特征维度)

        # 拆分 xyz 和其他特征
        xyz = self.point_data[:, :, :, :3]  # shape: (B, O, N, 3)
        fea = self.point_data[:, :, :, 3:]  # shape: (B, O, N, 3)

        # 全局 mean/std，只对 xyz 坐标计算
        xyz_flat = xyz.reshape(-1, 3)
        self.all_points_mean = xyz_flat.mean(axis=0).reshape(1, 1, 1, 3)
        self.all_points_std = xyz_flat.std(axis=0).reshape(1, 1, 1, 3)

        # 标准化 xyz 坐标
        xyz = (xyz - self.all_points_mean) / self.all_points_std

        # 将其他特征归一化到 [-1, 1]（假设原始在 [0, 1]）
        # fea = (fea - 0.5) * 2.0

        # 合并 xyz 和 feature 回原 shape
        self.point_data = np.concatenate([xyz, fea], axis=-1)  # shape: (B, O, N, 6)

    def get_pc_stats(self):


        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def load_eeg(self):
        eeg_path = self.data_path + 'EEGdata/'
        eeg_data_all, eeg_data_all2 = [], []
        for sub in self.sub_list:
            if self.train:
                sub_eeg = np.load(f"{eeg_path}{sub}/{sub}_train_data_6s_100Hz.npy")
                sub_eeg2 = np.load(f"{eeg_path}{sub}/{sub}_train_data_1s_250Hz.npy")
            else:
                sub_eeg = np.load(f"{eeg_path}{sub}/{sub}_test_data_6s_100Hz.npy")
                sub_eeg2 = np.load(f"{eeg_path}{sub}/{sub}_test_data_1s_250Hz.npy")
            eeg_data_all.append(sub_eeg[np.newaxis])
            eeg_data_all2.append(sub_eeg2[np.newaxis])
        return np.concatenate(eeg_data_all, axis=0), np.concatenate(eeg_data_all2, axis=0)

    def __len__(self):
        num = 1
        for i in range(len(self.eeg_data.shape) - 2):
            num *= self.eeg_data.shape[i]
        return num

    def add_noise(self, eeg_data):
        stds = eeg_data.std(dim=1, keepdim=True)
        stds[torch.isnan(stds)] = 0
        noise = torch.randn_like(eeg_data) * stds * 0.2
        return eeg_data + noise

    def __getitem__(self, idx):
        sub_index = idx // (self.cls_num * self.obj_num * self.trails_num)
        sub_other = idx % (self.cls_num * self.obj_num * self.trails_num)
        cls_index = sub_other // (self.obj_num * self.trails_num)
        cls_other = sub_other % (self.obj_num * self.trails_num)
        obj_index = cls_other // self.trails_num
        obj_other = cls_other % self.trails_num

        name = self.name_list[cls_index, obj_index]

        if self.aug_data and np.random.rand() > 0.75:
            eeg_data = np.mean(self.eeg_data[sub_index, cls_index, obj_index, :], axis=0)
        else:
            eeg_data = self.eeg_data[sub_index, cls_index, obj_index, obj_other]

        if self.aug_data and np.random.rand() > 0.4:
            eeg_data_new = self.add_noise(torch.from_numpy(eeg_data))
        else:
            eeg_data_new = torch.from_numpy(eeg_data)

        if self.aug_data and np.random.rand() > 0.75:
            eeg_data2 = np.mean(self.eeg_data2[sub_index, cls_index, obj_index, :], axis=0)
        else:
            eeg_data2 = self.eeg_data2[sub_index, cls_index, obj_index, obj_other]

        if self.aug_data and np.random.rand() > 0.4:
            eeg_data2_new = self.add_noise(torch.from_numpy(eeg_data2))
        else:
            eeg_data2_new = torch.from_numpy(eeg_data2)

        point = self.point_data[cls_index, obj_index]
        m, s = self.get_pc_stats()

        clip_key = name[3:]
        return {
            'name': name,
            'eeg_data': eeg_data_new,
            'eeg_data2': eeg_data2_new,
            'point_cloud': torch.from_numpy(point),
            'txt_fea': self.clip_features[clip_key]['text'],
            'color_video_fea': self.clip_features[clip_key]['video'],
            'color_point_fea': self.clip_features[clip_key]['point'],
            "mean": m,
            "std": s,
        }


# from torch.utils.data import Dataset
# import os
# import numpy as np
# import torch
# import open3d as o3d
# import pandas as pd
#
# class AllDataFeatureTwoEEG(Dataset):
#
#     def __init__(self, data_path, sub_list, train=True, time_len=250, test_mean=True, aug_data=False, point_path=''):
#         self.data_path = data_path
#         self.sub_list = sub_list
#         self.train = train
#         self.time_len = time_len
#         self.test_mean = test_mean
#         self.aug_data = aug_data
#
#         # ------划分视频训练与测试集------
#         all_name_list = sorted(os.listdir(self.data_path + 'video_new/'))
#         self.name_list = []
#
#         remove_name = ['08', '09'] if train else ['00', '01', '02', '03', '04', '05', '06', '07']
#
#         for name in all_name_list:
#             if name.endswith('.mp4') and name[-6:-4] not in remove_name:
#                 self.name_list.append(name[:-4])
#
#         self.name_list = np.array(self.name_list).reshape(72, -1)
#         # print("-----", self.name_list.shape) # (72, 8) (72, 2)
#         # ------划分视频训练与测试集------
#
#
#         self.point_data = np.zeros((self.name_list.shape[0], self.name_list.shape[1], 8192, 6))
#         point_path = point_path or self.data_path + 'point_cloud_simple/'
#
#         for ii in range(self.name_list.shape[0]):
#             for jj in range(self.name_list.shape[1]):
#                 self.point_data[ii, jj] = self.pc_norm(np.load(point_path + self.name_list[ii][jj][3:] + '.npy'))
#
#         self.eeg_data, self.eeg_data2 = self.load_eeg()
#
#         if not self.train:
#             if self.test_mean:
#                 self.eeg_data = np.mean(self.eeg_data, axis=3, keepdims=True)
#                 self.eeg_data2 = np.mean(self.eeg_data2, axis=3, keepdims=True)
#                 self.obj_num, self.trails_num = 2, 1
#             else:
#                 self.obj_num, self.trails_num = 2, 4
#         else:
#             self.obj_num, self.trails_num = 8, 2
#
#         self.cls_num = 72
#         self.clip_features = torch.load(self.data_path + 'clip_feature.pth')
#         for key in self.clip_features.keys():
#             self.clip_features[key]['point'] /= 8.0
#
#     def pc_norm(self, pc):
#         xyz = pc[:, :3]
#         other_feature = pc[:, 3:]
#         centroid = np.mean(xyz, axis=0)
#         xyz = xyz - centroid
#         m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
#         xyz = xyz / m
#         other_feature = (other_feature - 0.5) * 2
#         return np.concatenate((xyz, other_feature), axis=1)
#
#     def load_eeg(self):
#         eeg_path = self.data_path + 'EEGdata/'
#         eeg_data_all, eeg_data_all2 = [], []
#         for sub in self.sub_list:
#             if self.train:
#                 sub_eeg = np.load(f"{eeg_path}{sub}/{sub}_train_data_6s_100Hz.npy")
#                 sub_eeg2 = np.load(f"{eeg_path}{sub}/{sub}_train_data_1s_250Hz.npy")
#             else:
#                 sub_eeg = np.load(f"{eeg_path}{sub}/{sub}_test_data_6s_100Hz.npy")
#                 sub_eeg2 = np.load(f"{eeg_path}{sub}/{sub}_test_data_1s_250Hz.npy")
#             eeg_data_all.append(sub_eeg[np.newaxis])
#             eeg_data_all2.append(sub_eeg2[np.newaxis])
#         return np.concatenate(eeg_data_all, axis=0), np.concatenate(eeg_data_all2, axis=0)
#
#     def __len__(self):
#         num = 1
#         for i in range(len(self.eeg_data.shape) - 2):
#             num *= self.eeg_data.shape[i]
#         return num
#
#     def add_noise(self, eeg_data):
#         stds = eeg_data.std(dim=1, keepdim=True)
#         stds[torch.isnan(stds)] = 0
#         noise = torch.randn_like(eeg_data) * stds * 0.2
#         return eeg_data + noise
#
#     def __getitem__(self, idx):
#         sub_index = idx // (self.cls_num * self.obj_num * self.trails_num)
#         sub_other = idx % (self.cls_num * self.obj_num * self.trails_num)
#         cls_index = sub_other // (self.obj_num * self.trails_num)
#         cls_other = sub_other % (self.obj_num * self.trails_num)
#         obj_index = cls_other // self.trails_num
#         obj_other = cls_other % self.trails_num
#
#         name = self.name_list[cls_index, obj_index]
#
#         if self.aug_data and np.random.rand() > 0.75:
#             eeg_data = np.mean(self.eeg_data[sub_index, cls_index, obj_index, :], axis=0)
#         else:
#             eeg_data = self.eeg_data[sub_index, cls_index, obj_index, obj_other]
#
#         if self.aug_data and np.random.rand() > 0.4:
#             eeg_data_new = self.add_noise(torch.from_numpy(eeg_data))
#         else:
#             eeg_data_new = torch.from_numpy(eeg_data)
#
#         if self.aug_data and np.random.rand() > 0.75:
#             eeg_data2 = np.mean(self.eeg_data2[sub_index, cls_index, obj_index, :], axis=0)
#         else:
#             eeg_data2 = self.eeg_data2[sub_index, cls_index, obj_index, obj_other]
#
#         if self.aug_data and np.random.rand() > 0.4:
#             eeg_data2_new = self.add_noise(torch.from_numpy(eeg_data2))
#         else:
#             eeg_data2_new = torch.from_numpy(eeg_data2)
#
#         point = self.point_data[cls_index, obj_index]
#
#         clip_key = name[3:]
#         return {
#             'name': name,
#             'eeg_data': eeg_data_new,
#             'eeg_data2': eeg_data2_new,
#             'point_cloud': torch.from_numpy(point),
#             'txt_fea': self.clip_features[clip_key]['text'],
#             'color_video_fea': self.clip_features[clip_key]['video'],
#             'color_point_fea': self.clip_features[clip_key]['point'],
#         }
