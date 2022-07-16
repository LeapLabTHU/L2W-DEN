import os
import glob
import time
import argparse


model_names = ['MSDNet']

arg_parser = argparse.ArgumentParser(description='MSDNet Image classification')


# experiment related
exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--train_url', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')
exp_group.add_argument('--resume', type=str, default=None)
exp_group.add_argument('--finetune_from', type=str, default=None)
exp_group.add_argument('--evalmode', default=None,
                       choices=['anytime', 'dynamic', 'both'],
                       help='which mode to evaluate')
exp_group.add_argument('--evaluate_from', default='', type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=100, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=None, type=int, help='random seed')
exp_group.add_argument('--round', default=1, type=int)


# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D',
                        choices=['cifar10', 'cifar100', 'ImageNet'],
                        help='data to work on')
data_group.add_argument('--data_url', metavar='DIR', default='/data/cx/data',
                        help='path to dataset (default: data)')
data_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
data_group.add_argument('--index_path', type=str, default='save/')
data_group.add_argument('--use_valid', action='store_true', default=False,
                        help='use validation set or not')
data_group.add_argument('--num_sample_valid', default=5000, type=int)


# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', type=str, default='RANet')
arch_group.add_argument('--arch_config', type=str, default='RANet')
arch_group.add_argument('--reduction', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')


# # MSDNet config
arch_group.add_argument('--nBlocks', type=int, default=1)
arch_group.add_argument('--nChannels', type=int, default=32)
arch_group.add_argument('--base', type=int,default=4)
arch_group.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'])
arch_group.add_argument('--step', type=int, default=1)
arch_group.add_argument('--growthRate', type=int, default=6)
arch_group.add_argument('--grFactor', default='1-2-4', type=str)
arch_group.add_argument('--prune', default='max', choices=['min', 'max'])
arch_group.add_argument('--bnFactor', default='1-2-4')
arch_group.add_argument('--bottleneck', default=True, type=bool)


# training related
optim_group = arg_parser.add_argument_group('optimization', 'optimization setting')
optim_group.add_argument('--epochs', default=300, type=int, metavar='N',
                         help='number of total epochs to run (default: 300)')
optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('-b', '--batch-size', default=64, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.1)')
optim_group.add_argument('--lr-type', default='multistep', type=str, metavar='T',
                        help='learning rate strategy (default: multistep)',
                        choices=['cosine', 'multistep'])
optim_group.add_argument('--decay-rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)')


# Multiprocess
dist_group = arg_parser.add_argument_group('multiprocess', 'multiprocess setting')
dist_group.add_argument('--world_size', default=1, type=int,
                    help='number of nodes for distributed training')
dist_group.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
dist_group.add_argument('--dist_url', default='tcp://127.0.0.1:29501', type=str,
                    help='url used to set up distributed training')
dist_group.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
dist_group.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
dist_group.add_argument('--multiprocessing_distributed', default=True,  action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


# ours
ours_group = arg_parser.add_argument_group('ours', 'our setting')
ours_group.add_argument('--target_p_index', default=20, type=int)
ours_group.add_argument('--meta_net_hidden_size', default=100, type=int)
ours_group.add_argument('--meta_net_num_layers', default=1, type=int)
ours_group.add_argument('--meta_interval', default=1, type=int)
ours_group.add_argument('--meta_lr', default=1e-5, type=float)
ours_group.add_argument('--meta_weight_decay', type=float, default=0.)
ours_group.add_argument('--epsilon', type=float)
ours_group.add_argument('--constraint_dimension', type=str)
ours_group.add_argument('--meta_net_input_type', type=str, choices=['loss', 'conf', 'both'])


args = arg_parser.parse_args()

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000
