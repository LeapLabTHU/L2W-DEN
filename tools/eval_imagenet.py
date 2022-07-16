import os
import sys
import time
import math
import random
import shutil
import warnings
import pandas as pd
from collections import OrderedDict
sys.path.append(".")
sys.path.append("./tools/")

import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from args_msdnet import args
from models.op_counter import measure_model
from adaptive_inference import dynamic_evaluate
from utils import AverageMeter, accuracy, Logger, save_checkpoint, adjust_learning_rate, get_free_port


def main():

    global args

    if args.seed is not None:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) # for multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # # you need to give the save dir (args.train_url)
    args.train_url = args.train_url
    os.makedirs(args.train_url, exist_ok=True)
    logger = Logger(args.train_url + '/screen_output_eval.txt')
    args.print_custom = logger.log

    IM_SIZE = 224
    model = getattr(models, args.arch)(args)  # for MSDNet
    args.num_exits = len(model.classifier)  # MSDNet: 5
    n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)    
    n_flops_m = [f/1e6 for f in n_flops]
    np.savetxt(os.path.join(args.train_url, 'flops.txt'), n_flops_m)
    torch.save(n_flops, os.path.join(args.train_url, 'flops.pth'))
    del(model)

    # multiprocessing_distributed training
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size

    args.port = get_free_port()

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    assert args.multiprocessing_distributed
    assert args.distributed
    args.ngpus_per_node = ngpus_per_node
    args.gpu = gpu

    ip = '127.0.0.1'
    port = args.port
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=f'tcp://{ip}:{port}',
                            world_size=args.world_size, rank=args.rank)


    # create model
    model = getattr(models, args.arch)(args)  # MSDNet


    # DistributedDataParallel
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # load the checkpoint
    args.print_custom(args.evaluate_from)
    args.print_custom("=> loading checkpoint '{}'".format(args.evaluate_from))
    checkpoint = torch.load(args.evaluate_from, map_location='cpu')
    model_state_dict = checkpoint['state_dict']
    model.module.load_state_dict(model_state_dict)

    args.print_custom("=> loaded checkpoint '{}'".format(args.evaluate_from))


    # Data loading code
    traindir = os.path.join(f'{args.data_url}', 'train')
    valdir = os.path.join(f'{args.data_url}', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_set = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
    val_set = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    
    if os.path.exists(os.path.join(args.index_path, 'index.pth')):
        print('!!!!!! Load train_set_index !!!!!!')
        train_set_index = torch.load(os.path.join(args.index_path, 'index.pth'))
    else:
        if not os.path.exists(args.index_path):
            os.makedirs(args.index_path)
        print('!!!!!! Save train_set_index !!!!!!')
        train_set_index = torch.randperm(len(train_set))
        torch.save(train_set_index, os.path.join(args.index_path, 'index.pth'))

    num_sample_valid = 50000

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            train_set_index[-num_sample_valid:]),
        num_workers=args.workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # Dynamic evaluate
    if args.evalmode == 'anytime':
        validate(test_loader, model, criterion, args, set_name='test_eval')
    elif args.evalmode == 'dynamic':
        dynamic_evaluate(model, test_loader, val_loader, 'dynamic_test_eval.txt', args)

    return


def validate(loader, model, criterion, args, set_name):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    model.eval()
    
    top1, top5, losses = [], [], []
    for i in range(args.num_exits):
        losses.append(AverageMeter('Loss', ':.4e'))
        top1.append(AverageMeter('Acc@1', ':6.2f'))
        top5.append(AverageMeter('Acc@5', ':6.2f'))
        
    end = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)

            output = model(images)
            if not isinstance(output, list):
                output = [output]                

            for j in range(args.num_exits):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                loss = criterion(output[j], target)

                dist.all_reduce(prec1)
                prec1 /= args.world_size
                dist.all_reduce(prec5)
                prec5 /= args.world_size
                dist.all_reduce(loss)
                loss /= args.world_size

                top1[j].update(prec1.item(), images.size(0))
                top5[j].update(prec5.item(), images.size(0))
                losses[j].update(loss.item(), images.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i % args.print_freq == 0) or (i == len(loader) - 1):
                args.print_custom('Epoch: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f}({data_time.avg:.3f})\t'
                    'Loss1 {loss1.val:.4f}({loss1.avg:.4f})\t'
                    'Loss2 {loss2.val:.4f}({loss2.avg:.4f})\t'
                    'Loss3 {loss3.val:.4f}({loss3.avg:.4f})\t'
                    'Loss4 {loss4.val:.4f}({loss4.avg:.4f})\t'
                    'Loss5 {loss5.val:.4f}({loss5.avg:.4f})\t'
                    'Acc@1 {top1.val:.4f}({top1.avg:.4f})\t'
                    'Acc@5 {top5.val:.4f}({top5.avg:.4f})'
                    .format(
                            i, len(loader),
                            batch_time=batch_time, data_time=data_time,
                            loss1=losses[0], loss2=losses[1], loss3=losses[2], loss4=losses[3], loss5=losses[4],
                            top1=top1[-1], top5=top5[-1]))
        ce_loss = []
        acc1_exits = []
        acc5_exits = []
        for j in range(args.num_exits):
            ce_loss.append(losses[j].avg)
            acc1_exits.append(top1[j].avg)
            acc5_exits.append(top5[j].avg)

        ce_loss_arr = np.array(ce_loss)[..., np.newaxis]
        acc1_exits_arr = np.array(acc1_exits)[..., np.newaxis]
        acc5_exits_arr = np.array(acc5_exits)[..., np.newaxis]
        array_to_save = np.concatenate((ce_loss_arr, acc1_exits_arr, acc5_exits_arr), axis=1)

        log_file = f'{args.train_url}/AnytimeResults_on_all_{set_name}.csv'
        np.savetxt(log_file, array_to_save, fmt='%6.4f', delimiter=",")
        return ce_loss, acc1_exits, acc5_exits


if __name__ == '__main__':
    main()