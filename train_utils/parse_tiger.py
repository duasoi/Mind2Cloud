import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='ShapeNetCore.v2.PC15k/')
    parser.add_argument('--category', default='chair')

    parser.add_argument('--bs', type=int, default=64, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    '''model'''
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    #params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")




    '''eval'''
    parser.add_argument('--saveIter', default=100, help='unit: epoch')
    parser.add_argument('--diagIter', default=50, help='unit: epoch')
    parser.add_argument('--vizIter', default=50, help='unit: epoch')
    parser.add_argument('--print_freq', default=50, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')


    opt = parser.parse_args()

    return opt