import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data1/hxf/neuro-3D-main/EEG_3Datasets/')
    parser.add_argument('--model_save_path', type=str, default='model/')
    parser.add_argument('--sub', type=str, default='sub01')
    # parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--pretrain_model', type=str, default='')


    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--log_step_freq', type=int, default=10)
    parser.add_argument('--checkpoint_freq', type=int, default=5000)
    parser.add_argument('--test_freq', type=int, default=5000)

    parser.add_argument('--sample_num_points', type=int, default=2048, help='Number of points to sample in point cloud')

    parser.add_argument('--mixed_precision', type=str, default='fp16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--beta_start', type=float, default=1e-5)
    parser.add_argument('--beta_end', type=float, default=8e-3)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--point_cloud_model_embed_dim', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=1027)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--generation_type', type=str, default='shape', choices=['shape', 'color'])
    parser.add_argument('--model_type', type=str, default='', choices=['', 'dynamic', 'static', 'concat', 'NoDe'])

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--checkpoint_resume', type=str, default='')
    parser.add_argument('--clip_grad_norm', type=float, default=50.0)
    parser.add_argument('--ply_point_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')

### ---------------------------------------------------------Tiger------------------------------------------------------------------------------------------

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataroot', default='ShapeNetCore.v2.PC15k/')
    # parser.add_argument('--category', default='chair')

    parser.add_argument('--bs', type=int, default=64, help='input batch size')
    # parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    '''model'''
    parser.add_argument('--beta_start_tiger', default=0.0001)
    parser.add_argument('--beta_end_tiger', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    # params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr_tiger', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    # parser.add_argument('--model', default='', help="path to model (to continue training)")


    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')


    parser.add_argument('--saveIter', default=100, help='unit: epoch')
    parser.add_argument('--diagIter', default=50, help='unit: epoch')
    parser.add_argument('--vizIter', default=50, help='unit: epoch')
    parser.add_argument('--print_freq', default=50, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')


    opt = parser.parse_args()
    return opt