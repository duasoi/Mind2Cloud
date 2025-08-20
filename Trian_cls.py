import os
import sys
import torch
import numpy as np

import datetime
import logging
# import provider
import importlib
import shutil
import argparse
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

sys.path.append('.')
from PointClassify.pointnet_cls import get_model, get_loss
import open3d as o3d
import natsort
import copy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=72, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


class AllDataFeature2(Dataset):

    def __init__(self, data_path, train=True):
        self.data_path = data_path
        all_name_list = sorted(os.listdir(self.data_path + 'video_new/'))
        remove_name = ['08', '09']
        name_to_cls_dir = {}
        for name in all_name_list:
            if name[-4:] != '.mp4':
                continue
            if name[-6:-4] in remove_name:
                continue
            name_to_cls_dir[name[3:-6]] = int(name[:2]) - 1
        name_to_cls_dir['crucifix_'] = 72
        name_to_cls_dir['blazer_'] = 73

        point_path1 = self.data_path + 'point_cloud/'
        point_path2 = self.data_path + 'point_cloud2/'
        point_name_list = os.listdir(point_path1) + os.listdir(point_path2)
        point_name_list = natsort.natsorted(point_name_list)
        # import pdb;pdb.set_trace()
        self.point_list, self.label_list = [], []
        self.name_list = []
        # import pdb;pdb.set_trace()
        for name in point_name_list:
            if '.npy' not in name:
                continue
            if train:
                if name[-6:-4] in remove_name:
                    continue
            else:
                if name[-6:-4] not in remove_name:
                    continue
            if name[-6:-4] >= '10':
                point_name_one = point_path2 + name
            else:
                point_name_one = point_path1 + name
            point_data_one = self.pc_norm(np.load(point_name_one))
            self.point_list.append(point_data_one[np.newaxis, :, :])
            self.label_list.append(name_to_cls_dir[name[:-6]])
            self.name_list.append(name[:-4])
        self.label_list = np.array(self.label_list)
        self.point_list = np.concatenate(self.point_list, axis=0)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        other_feature = (other_feature - 0.5) * 2
        # other_feature = (other_feature - other_feature.mean()) / other_feature.std()
        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc

    def __len__(self, ):
        return self.point_list.shape[0]

    def __getitem__(self, idx):
        label = self.label_list[idx]
        point = self.point_list[idx]
        name = self.name_list[idx]
        return {'name': name, 'cls_label': label, 'point_cloud': torch.from_numpy(point)}


def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    other_feature = (other_feature - 0.5) * 2
    # other_feature = (other_feature - other_feature.mean()) / other_feature.std()
    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(args, model, loader, num_class=72, k=1):
    mean_correct, tok_k_acc_mean = [], []
    class_acc = np.zeros((72, 3))
    classifier = model.eval()
    for j, batch_data in tqdm(enumerate(loader), total=len(loader)):

        points = batch_data['point_cloud'].float()[:, :, :3]
        target = batch_data['cls_label']

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred = pred.softmax(1)
        pred_choice = pred.data.max(1)[1]
        # import pdb;pdb.set_trace()
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

        topk_pred = pred.topk(k, dim=1)[1]
        correct = topk_pred.eq(target.view(-1, 1).expand_as(topk_pred)).sum().item()
        tok_k_acc = correct / float(points.size(0))
        tok_k_acc_mean.append(tok_k_acc)

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    tok_k_acc_re = np.mean(tok_k_acc_mean)
    print(tok_k_acc_re)
    if k == 1:
        return instance_acc, class_acc
    else:
        return instance_acc, class_acc, tok_k_acc_re


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    exp_dir = ""
    os.makedirs(exp_dir, exist_ok=True)

    checkpoints_dir = exp_dir + 'checkpoints_new_74/'
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_dir = exp_dir + 'logs_new_74/'
    os.makedirs(log_dir, exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = ''

    train_dataset = AllDataFeature2(data_path, train=True)
    test_dataset = AllDataFeature2(data_path, train=False)
    print(f"train len:{train_dataset.__len__()}, test len:{test_dataset.__len__()}")

    # train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category

    classifier = get_model(num_class, normal_channel=args.use_normals)
    criterion = get_loss()
    classifier.apply(inplace_relu)
    # import pdb;pdb.set_trace()
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # try:
    #     checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     log_string('Use pretrain model')
    # except:
    #     log_string('No existing model, starting training from scratch...')
    #     start_epoch = 0
    start_epoch = 0
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, batch_data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = batch_data['point_cloud'].float()[:, :, :3]
            target = batch_data['cls_label']

            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(args, classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

