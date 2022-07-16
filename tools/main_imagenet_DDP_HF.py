from ffrecord.torch import Dataset, DataLoader
import pickle

import os
import sys
import math
import time
import copy
import shutil
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
sys.path.append(".")
sys.path.append("./tools/")

import models
from args_msdnet import args
from utils import AverageMeter, accuracy, Logger, save_checkpoint, adjust_learning_rate, adjust_meta_learning_rate, calc_target_probs, get_scores_string, get_num_string
from adaptive_inference import dynamic_evaluate
from models.op_counter import measure_model
from meta import MetaSGD

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets


best_prec1, best_epoch = 0.0, 0

def main():
    global args

    if args.seed is not None:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed) 
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True    
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    args.train_url = os.path.join(
        './output/',
        args.train_url,
        f'{args.data}_{args.arch}_eps{args.epochs}_bs{args.batch_size}_lr{args.lr}_nBlock{args.nBlocks}_stepmode{args.stepmode}_step{args.step}_base{args.base}_nChannels{args.nChannels}_mhidden{args.meta_net_hidden_size}_mlayers{args.meta_net_num_layers}_mintv{args.meta_interval}_mlr{args.meta_lr}_mwd{args.meta_weight_decay}_meta_inp_type_{args.meta_net_input_type}_constrdim_{args.constraint_dimension}_epsilon{args.epsilon:.2f}_targetp{args.target_p_index:02d}/',
        )
    os.makedirs(args.train_url, exist_ok=True)
    
    args.loss_weight_save_dir = f'{args.train_url}/loss_and_weights'
    os.makedirs(args.loss_weight_save_dir, exist_ok=True)
    if args.evalmode is not None:
        logger = Logger(args.train_url + '/screen_output_eval.txt')
    elif args.resume is not None:
        logger = Logger(args.train_url + '/screen_output_resume.txt')
    else:
        logger = Logger(args.train_url + '/screen_output.txt')
    args.print_custom = logger.log
    args.print_custom(args.train_url)

    IM_SIZE = 224

    model = getattr(models, args.arch)(args)
    args.num_exits = model.nBlocks
    n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)    
    n_flops_m = [f/1e6 for f in n_flops]
    np.savetxt(os.path.join(args.train_url, 'flops.txt'), n_flops_m)
    torch.save(n_flops, os.path.join(args.train_url, 'flops.pth'))
    del(model)

    mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(torch.cuda.device_count(), args))  # ngpus_per_node = torch.cuda.device_count() == 8


def main_worker(gpu, ngpus_per_node, args):
    """
    for multi-nodes training, the gpu is still 0-7, but each node has 0-7
    """
    global best_prec1, best_epoch
    args.gpu = gpu
    local_rank = gpu

    # Multiprocessing
    ip = os.environ['MASTER_IP']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])               # number of gpu servers
    rank = int(os.environ['RANK'])                      # rank of current server
    gpus = torch.cuda.device_count()                    # number of gpu of each server
    ngpus_per_node = torch.cuda.device_count()          # number of gpu of each server
    args.is_main_proc = (rank * gpus + local_rank == 0)
    args.world_size = hosts * gpus
    dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts*gpus, rank=rank*gpus+local_rank)


    model = getattr(models, args.arch)(args)

    
    if args.meta_net_input_type in ['loss', 'conf']:
        meta_net = getattr(models, 'MLP_tanh')(input_size=args.num_exits, 
                                      hidden_size=args.meta_net_hidden_size, 
                                      num_layers=args.meta_net_num_layers,
                                      output_size=args.num_exits)
    else:
        meta_net = getattr(models, 'MLP_tanh')(input_size=args.num_exits*2, 
                                      hidden_size=args.meta_net_hidden_size, 
                                      num_layers=args.meta_net_num_layers,
                                      output_size=args.num_exits)


    """ model DDP"""
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    meta_net.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    meta_net = torch.nn.parallel.DistributedDataParallel(meta_net, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
    

    if args.resume is not None:
        args.print_custom(args.resume)
        if os.path.isfile(args.resume):
            args.print_custom("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location='cpu')

            args.start_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            best_prec1 = checkpoint['best_prec1']
            is_best = checkpoint['is_best']
            
            meta_net.module.load_state_dict(checkpoint['meta_state_dict'])
            meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
            
            args.print_custom("=> loaded checkpoint '{}'".format(args.resume))
        else:
            args.print_custom("=> NO checkpoint at '{}'".format(args.resume))


    cudnn.benchmark = True


    # ImageNet Dataloader
    train_loader, _, test_loader, train_sampler = load_imagenet_data(args)


    scores = [f'epoch\tlr\tmeta_lr'
        + get_scores_string(num_exits=args.num_exits, prefix="\tloss_train_", suffix="_onAll")
        + get_scores_string(num_exits=args.num_exits, prefix="\tacc1_train_", suffix="_onAll")
        + f'\tmeta_loss_train'
        + get_scores_string(num_exits=args.num_exits, prefix="\tloss_val_", suffix="_onAll")
        + get_scores_string(num_exits=args.num_exits, prefix="\tacc1_val_", suffix="_onAll")
        ]
    if args.gpu == 0:
        if not os.path.exists(args.train_url + '/scores_all.tsv'):
            with open(args.train_url + '/scores_all.tsv', "a") as f:
                f.write(
                    f'epoch\tlr\tmeta_lr'
                    + get_scores_string(num_exits=args.num_exits, prefix="\tloss_train_", suffix="_onAll")
                    + get_scores_string(num_exits=args.num_exits, prefix="\tacc1_train_", suffix="_onAll")
                    + f'\tmeta_loss_train'
                    + get_scores_string(num_exits=args.num_exits, prefix="\tloss_val_", suffix="_onAll")
                    + get_scores_string(num_exits=args.num_exits, prefix="\tacc1_val_", suffix="_onAll")
                    +  '\n'
                )
        
    probs = calc_target_probs(args)
    target_probs = probs[args.target_p_index-1]
    args.print_custom(target_probs)


    if args.evalmode is not None:
        state_dict = torch.load(args.evaluate_from, map_location='cpu')['state_dict']
        model.module.load_state_dict(state_dict)

        validate(test_loader, model, criterion, set_name='test_eval', args=args)
        dynamic_evaluate(model, test_loader, train_loader, filename='dynamic_test_eval.txt', args=args)
        return


    # auto resume
    resume_dir = os.path.join(args.train_url, "save_models", "checkpoint.pth.tar")
    if os.path.exists(resume_dir):
        args.print_custom(f'[INFO] resume dir: {resume_dir}')
        ckpt = torch.load(resume_dir, map_location='cpu')
        args.start_epoch = ckpt['epoch']
        model.module.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_prec1 = ckpt['best_prec1']
        is_best = ckpt['is_best']
        meta_net.module.load_state_dict(ckpt['meta_state_dict'])
        meta_optimizer.load_state_dict(ckpt['meta_optimizer'])
        args.print_custom(f'[INFO] Auto Resume from {resume_dir}, from  finished epoch {args.start_epoch}, with acc {best_prec1}.')


    t_start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.world_size > 1:
            train_sampler.set_epoch(epoch)

        ce_loss_train, acc1_exits_train, _, meta_loss_train, lr, meta_lr, all_losses, all_confidences, all_weights = train(train_loader, model, meta_net, criterion, optimizer, meta_optimizer, epoch, target_probs, args)
        
        if (args.gpu == 0) and ((epoch % 5 == 0) or (epoch >= args.epochs-20)): 
            all_losses = pd.DataFrame(all_losses.cpu().numpy())
            all_weights = pd.DataFrame(all_weights.cpu().numpy())
            all_confidences = pd.DataFrame(all_confidences.cpu().numpy())

            np.savetxt(os.path.join(f'{args.loss_weight_save_dir}/loss_epoch{epoch}.txt'), all_losses)
            np.savetxt(os.path.join(f'{args.loss_weight_save_dir}/confidence_epoch{epoch}.txt'), all_confidences)
            np.savetxt(os.path.join(f'{args.loss_weight_save_dir}/weights_epoch{epoch}.txt'), all_weights)
        
        ce_loss_val, acc1_exits_val, _ = validate(test_loader, model, criterion, set_name='test', args=args)

        scores.append(
            f"{epoch}\t{lr:.7f}\t{meta_lr:.7f}" + get_num_string(ce_loss_train) + get_num_string(acc1_exits_train) + f"\t{meta_loss_train:.4f}" + get_num_string(ce_loss_val) + get_num_string(acc1_exits_val)
        )
        if args.gpu == 0:
            with open(args.train_url + '/scores_all.tsv', "a") as f:
                f.write(
                    f"{epoch}\t{lr:.7f}\t{meta_lr:.7f}" + get_num_string(ce_loss_train) + get_num_string(acc1_exits_train) + f"\t{meta_loss_train:.4f}" + get_num_string(ce_loss_val) + get_num_string(acc1_exits_val) + '\n'
                )

        val_prec1 = acc1_exits_val[-1]
        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            args.print_custom('Best var_prec1 {}'.format(best_prec1))

        if args.gpu == 0:
            model_filename = 'checkpoint.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
                'is_best': is_best,
                'meta_state_dict': meta_net.module.state_dict(),
                'meta_optimizer': meta_optimizer.state_dict(),
            }, args, is_best, model_filename, scores)

            t_curr = time.time()
            eta_total = (t_curr - t_start) / (epoch + 1 - args.start_epoch) * (args.epochs - epoch - 1)
            eta_hour = int(eta_total // 3600)
            eta_min = int((eta_total - eta_hour * 3600) // 60)
            eta_sec = int(eta_total - eta_hour * 3600 - eta_min * 60)
            args.print_custom(f'[INFO] Finished epoch:{epoch:02d};  ETA {eta_hour:02d} h {eta_min:02d} m {eta_sec:02d} s')

    args.print_custom('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))

    ### Test the final model
    args.print_custom('********** Final prediction results **********')
    validate(test_loader, model, criterion, set_name=f'test_final', args=args)
    dynamic_evaluate(model, test_loader, train_loader, filename=f'dynamic_test_final.txt', args=args)

    return 


def train(train_loader, model, meta_net, criterion, optimizer, meta_optimizer, epoch, target_probs, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    meta_losses = AverageMeter('MetaLoss', ':.4e')
    top1, top5, all_losses = [], [], []
    for i in range(args.num_exits):
        all_losses.append(AverageMeter('Loss', ':.4e'))
        top1.append(AverageMeter('Acc@1', ':6.2f'))
        top5.append(AverageMeter('Acc@5', ':6.2f'))

    # switch to train mode
    model.train()
    running_lr, running_meta_lr = None, None
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i, nBatch=len(train_loader))
        meta_lr = adjust_meta_learning_rate(meta_optimizer, epoch, args, batch=i, nBatch=len(train_loader))
        
        if running_lr is None:
            running_lr = lr
        if running_meta_lr is None:
            running_meta_lr = meta_lr

        images_p1, images_p2 = images.chunk(2, dim=0)
        target_p1, target_p2 = target.chunk(2, dim=0)
        data_time.update(time.time() - end)


        ###################################################
        ## part 1: images_p1 as train, images_p2 as meta ##
        ###################################################

        if i % args.meta_interval == 0:
            pseudo_net = getattr(models, args.arch)(args).cuda(args.gpu)
            pseudo_net.load_state_dict(model.module.state_dict())
            pseudo_net.train()

            pseudo_outputs = pseudo_net(images_p1)
            if not isinstance(pseudo_outputs, list):
                pseudo_outputs = [pseudo_outputs]

            if args.meta_net_input_type == 'loss':
                for j in range(args.num_exits):
                    pseudo_loss_vector = F.cross_entropy(pseudo_outputs[j], target_p1, reduction='none')
                    if j==0:
                        pseudo_losses = pseudo_loss_vector.unsqueeze(1)
                    else:
                        pseudo_losses = torch.cat((pseudo_losses, pseudo_loss_vector.unsqueeze(1)), dim=1)
                input_of_meta = pseudo_losses
            else:
                for j in range(args.num_exits):
                    pseudo_loss_vector = F.cross_entropy(pseudo_outputs[j], target_p1, reduction='none')
                    confidence = F.softmax(pseudo_outputs[j], dim=1)
                    confidence, _ = confidence.max(dim=1, keepdim=False)
                    if j==0:
                        pseudo_losses = pseudo_loss_vector.unsqueeze(1)
                        confidences = confidence.unsqueeze(1)
                    else:
                        pseudo_losses = torch.cat((pseudo_losses, pseudo_loss_vector.unsqueeze(1)), dim=1)
                        confidences = torch.cat((confidences, confidence.unsqueeze(1)), dim=1)
                input_of_meta = confidences if args.meta_net_input_type == 'conf' else torch.cat((pseudo_losses, confidences), dim=1)

            pseudo_weight = meta_net(input_of_meta.detach())
            if args.constraint_dimension == 'row':
                pseudo_weight = pseudo_weight - torch.mean(pseudo_weight, 0 ,keepdim=True)  # 1x5
            elif args.constraint_dimension == 'col':
                pseudo_weight = pseudo_weight - torch.mean(pseudo_weight, 1 ,keepdim=True)  # Bx1
            elif args.constraint_dimension == 'mat':
                pseudo_weight = pseudo_weight - torch.mean(pseudo_weight)  # 1
            else:
                args.print_custom('Wrong mode')
                assert(0==1)
            pseudo_weight = torch.ones(pseudo_weight.shape).to(pseudo_weight.device) + args.epsilon * pseudo_weight

            pseudo_loss_multi_exits = torch.sum(torch.mean(pseudo_weight * pseudo_losses, 0))

            pseudo_grads = torch.autograd.grad(pseudo_loss_multi_exits, pseudo_net.parameters(), create_graph=True)

            pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
            pseudo_optimizer.load_state_dict(optimizer.state_dict())
            pseudo_optimizer.meta_step(pseudo_grads)

            del pseudo_grads

            meta_outputs = pseudo_net(images_p2)
            if not isinstance(meta_outputs, list):
                meta_outputs = [meta_outputs]
            
            used_index = []
            meta_loss = 0.0
            for j in range(args.num_exits):
                with torch.no_grad():
                    confidence_target = F.softmax(meta_outputs[j], dim=1)  
                    max_preds_target, _ = confidence_target.max(dim=1, keepdim=False)  
                    _, sorted_idx = max_preds_target.sort(dim=0, descending=True)  
                    n_target = sorted_idx.shape[0]
                    
                    if j == 0:
                        selected_index = sorted_idx[: math.floor(n_target * target_probs[j])]
                        selected_index = selected_index.tolist()
                        used_index.extend(selected_index)
                    elif j < args.num_exits - 1:
                        filter_set = set(used_index)
                        unused_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                        selected_index = unused_index[: math.floor(n_target * target_probs[j])]  
                        used_index.extend(selected_index)
                    else:
                        filter_set = set(used_index)
                        selected_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                if len(selected_index) > 0:
                    meta_loss += F.cross_entropy(meta_outputs[j][selected_index], target_p2[selected_index].long(), reduction='mean')
            meta_losses.update(meta_loss.item(), images_p2.size(0))

            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

        outputs = model(images_p1)
        if not isinstance(outputs, list):
            outputs = [outputs]

        for j in range(args.num_exits):
            loss_vector = F.cross_entropy(outputs[j], target_p1, reduction='none')
            confidence = F.softmax(outputs[j], dim=1)
            confidence, _ = confidence.max(dim=1, keepdim=False)
            if j==0:
                losses = loss_vector.unsqueeze(1)
                confidences = confidence.unsqueeze(1)
            else:
                losses = torch.cat((losses, loss_vector.unsqueeze(1)), dim=1)
                confidences = torch.cat((confidences, confidence.unsqueeze(1)), dim=1)

        if args.meta_net_input_type == 'loss':
            input_of_meta = losses
        elif args.meta_net_input_type == 'conf':
            input_of_meta = confidences
        else:
            input_of_meta = torch.cat((losses, confidences), dim=1)

        with torch.no_grad():
            weight = meta_net(input_of_meta)
            if args.constraint_dimension == 'row':
                weight = weight - torch.mean(weight, 0 ,keepdim=True)  # 1x5
            elif args.constraint_dimension == 'col':
                weight = weight - torch.mean(weight, 1 ,keepdim=True)  # Bx1
            elif args.constraint_dimension == 'mat':
                weight = weight - torch.mean(weight)                   # 1
            else:
                args.print_custom('Wrong mode')
                assert(0==1)
            weight = torch.ones(weight.shape).to(weight.device) + args.epsilon * weight
            if i == 0:
                all_losses_record = losses
                all_confidences_record = confidences
                all_weights = weight
            else:
                all_losses_record = torch.cat((all_losses_record, losses), dim=0)
                all_confidences_record = torch.cat((all_confidences_record, confidences), dim=0)
                all_weights = torch.cat((all_weights, weight), dim=0)
        
        loss_multi_exits = torch.mean(weight * losses, 0)
        
        for j in range(args.num_exits):
            all_losses[j].update(loss_multi_exits[j].item(), images_p1.size(0))
            prec1, prec5 = accuracy(outputs[j].data, target_p1, topk=(1, 5))
            top1[j].update(prec1.item(), images_p1.size(0))
            top5[j].update(prec5.item(), images_p1.size(0))
        
        loss_multi_exits = torch.sum(loss_multi_exits)

        optimizer.zero_grad()
        loss_multi_exits.backward()
        optimizer.step()


        ###################################################
        ## part 2: images_p2 as train, images_p1 as meta ##
        ###################################################

        if i % args.meta_interval == 0:
            pseudo_net = getattr(models, args.arch)(args).cuda(args.gpu)
            pseudo_net.load_state_dict(model.module.state_dict())
            pseudo_net.train()

            pseudo_outputs = pseudo_net(images_p2)
            if not isinstance(pseudo_outputs, list):
                pseudo_outputs = [pseudo_outputs]

            if args.meta_net_input_type == 'loss':
                for j in range(args.num_exits):
                    pseudo_loss_vector = F.cross_entropy(pseudo_outputs[j], target_p2, reduction='none')
                    if j==0:
                        pseudo_losses = pseudo_loss_vector.unsqueeze(1)
                    else:
                        pseudo_losses = torch.cat((pseudo_losses, pseudo_loss_vector.unsqueeze(1)), dim=1)
                input_of_meta = pseudo_losses
            else:
                for j in range(args.num_exits):
                    pseudo_loss_vector = F.cross_entropy(pseudo_outputs[j], target_p2, reduction='none')
                    confidence = F.softmax(pseudo_outputs[j], dim=1)
                    confidence, _ = confidence.max(dim=1, keepdim=False)
                    if j==0:
                        pseudo_losses = pseudo_loss_vector.unsqueeze(1)
                        confidences = confidence.unsqueeze(1)
                    else:
                        pseudo_losses = torch.cat((pseudo_losses, pseudo_loss_vector.unsqueeze(1)), dim=1)
                        confidences = torch.cat((confidences, confidence.unsqueeze(1)), dim=1)
                input_of_meta = confidences if args.meta_net_input_type == 'conf' else torch.cat((pseudo_losses, confidences), dim=1)

            pseudo_weight = meta_net(input_of_meta.detach())  # TODO: .detach() or .data?
            if args.constraint_dimension == 'row':
                pseudo_weight = pseudo_weight - torch.mean(pseudo_weight, 0 ,keepdim=True)  # 1x5
            elif args.constraint_dimension == 'col':
                pseudo_weight = pseudo_weight - torch.mean(pseudo_weight, 1 ,keepdim=True)  # Bx1
            elif args.constraint_dimension == 'mat':
                pseudo_weight = pseudo_weight - torch.mean(pseudo_weight)  # 1
            else:
                args.print_custom('Wrong mode')
                assert(0==1)
            pseudo_weight = torch.ones(pseudo_weight.shape).to(pseudo_weight.device) + args.epsilon * pseudo_weight

            pseudo_loss_multi_exits = torch.sum(torch.mean(pseudo_weight * pseudo_losses, 0))

            pseudo_grads = torch.autograd.grad(pseudo_loss_multi_exits, pseudo_net.parameters(), create_graph=True)

            pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
            pseudo_optimizer.load_state_dict(optimizer.state_dict())
            pseudo_optimizer.meta_step(pseudo_grads)

            del pseudo_grads

            meta_outputs = pseudo_net(images_p1)
            if not isinstance(meta_outputs, list):
                meta_outputs = [meta_outputs]
            
            used_index = []
            meta_loss = 0.0
            for j in range(args.num_exits):
                with torch.no_grad():
                    confidence_target = F.softmax(meta_outputs[j], dim=1)  
                    max_preds_target, _ = confidence_target.max(dim=1, keepdim=False)  
                    _, sorted_idx = max_preds_target.sort(dim=0, descending=True)  
                    n_target = sorted_idx.shape[0]
                    
                    if j == 0:
                        selected_index = sorted_idx[: math.floor(n_target * target_probs[j])]
                        selected_index = selected_index.tolist()
                        used_index.extend(selected_index)
                    elif j < args.num_exits - 1:
                        filter_set = set(used_index)
                        unused_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                        selected_index = unused_index[: math.floor(n_target * target_probs[j])]  
                        used_index.extend(selected_index)
                    else:
                        filter_set = set(used_index)
                        selected_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                if len(selected_index) > 0:
                    meta_loss += F.cross_entropy(meta_outputs[j][selected_index], target_p1[selected_index].long(), reduction='mean')
            meta_losses.update(meta_loss.item(), images_p1.size(0))

            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

        outputs = model(images_p2)
        if not isinstance(outputs, list):
            outputs = [outputs]

        for j in range(args.num_exits):
            loss_vector = F.cross_entropy(outputs[j], target_p2, reduction='none')
            confidence = F.softmax(outputs[j], dim=1)
            confidence, _ = confidence.max(dim=1, keepdim=False)
            if j==0:
                losses = loss_vector.unsqueeze(1)
                confidences = confidence.unsqueeze(1)
            else:
                losses = torch.cat((losses, loss_vector.unsqueeze(1)), dim=1)
                confidences = torch.cat((confidences, confidence.unsqueeze(1)), dim=1)

        if args.meta_net_input_type == 'loss':
            input_of_meta = losses
        elif args.meta_net_input_type == 'conf':
            input_of_meta = confidences
        else:
            input_of_meta = torch.cat((losses, confidences), dim=1)

        with torch.no_grad():
            weight = meta_net(input_of_meta)
            if args.constraint_dimension == 'row':
                weight = weight - torch.mean(weight, 0 ,keepdim=True)  # 1x5
            elif args.constraint_dimension == 'col':
                weight = weight - torch.mean(weight, 1 ,keepdim=True)  # Bx1
            elif args.constraint_dimension == 'mat':
                weight = weight - torch.mean(weight)                   # 1
            else:
                args.print_custom('Wrong mode')
                assert(0==1)
            weight = torch.ones(weight.shape).to(weight.device) + args.epsilon * weight
            if i == 0:
                all_losses_record = losses
                all_confidences_record = confidences
                all_weights = weight
            else:
                all_losses_record = torch.cat((all_losses_record, losses), dim=0)
                all_confidences_record = torch.cat((all_confidences_record, confidences), dim=0)
                all_weights = torch.cat((all_weights, weight), dim=0)
        
        loss_multi_exits = torch.mean(weight * losses, 0)
        
        for j in range(args.num_exits):
            all_losses[j].update(loss_multi_exits[j].item(), images_p2.size(0))
            prec1, prec5 = accuracy(outputs[j].data, target_p2, topk=(1, 5))
            top1[j].update(prec1.item(), images_p2.size(0))
            top5[j].update(prec5.item(), images_p2.size(0))
        
        loss_multi_exits = torch.sum(loss_multi_exits)

        optimizer.zero_grad()
        loss_multi_exits.backward()
        optimizer.step()

        ###################################################
        ##                 end exchange                  ##
        ###################################################

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i % args.print_freq == 0) and (args.gpu == 0):
            args.print_custom('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MetaLoss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Acc@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                    epoch, i + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=all_losses[-1], meta_loss=meta_losses, top1=top1[-1], top5=top5[-1]))

    ce_loss = []
    acc1_exits = []
    acc5_exits = []
    for j in range(args.num_exits):
        ce_loss.append(all_losses[j].avg)
        acc1_exits.append(top1[j].avg)
        acc5_exits.append(top5[j].avg)

    return ce_loss, acc1_exits, acc5_exits, meta_losses.avg, running_lr, running_meta_lr, all_losses_record, all_confidences_record, all_weights


def validate(loader, model, criterion, set_name, args):
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
            if (args.gpu == 0) and ((i % args.print_freq == 0) or (i == len(loader) - 1)):
                args.print_custom('Epoch: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f}({data_time.avg:.3f})\t'
                    'Acc@1 {top1.val:.4f}({top1.avg:.4f})\t'
                    'Acc@5 {top5.val:.4f}({top5.avg:.4f})'
                    .format(
                            i, len(loader),
                            batch_time=batch_time, data_time=data_time,
                            top1=top1[-1], top5=top5[-1]))
                
        ce_loss = []
        acc1_exits = []
        acc5_exits = []
        for j in range(args.num_exits):
            ce_loss.append(losses[j].avg)
            acc1_exits.append(top1[j].avg)
            acc5_exits.append(top5[j].avg)
            
        df = pd.DataFrame({'ce_loss': ce_loss, 'acc1_exits': acc1_exits, 'acc5_exits':acc5_exits})

        log_file = f'{args.train_url}/AnytimeResults_on_all_{set_name}.csv'
        with open(log_file, "w") as f:
            df.to_csv(f)

        return ce_loss, acc1_exits, acc5_exits


def load_imagenet_data(args):
    """ ImageFolder Style """
    """traindir = os.path.join(f'{args.data_url}', 'train')
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

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, shuffle=(train_sampler==None))

    test_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)"""

    """ FireFLyerImageNet Style"""
    # Data loading code
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    train_dataset = FireFlyerImageNet('/public_dataset/1/ImageNet/train.ffr', transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True, shuffle=(train_sampler==None), drop_last=False)


    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    test_dataset = FireFlyerImageNet('/public_dataset/1/ImageNet/val.ffr', transform=test_transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    return train_loader, None, test_loader, train_sampler


class FireFlyerImageNet(Dataset):
    def __init__(self, fname, transform=None):
        super(FireFlyerImageNet, self).__init__(fname, check_data=True)
        self.transform = transform

    def process(self, indexes, data):
        samples = []

        for bytes_ in data:
            img, label = pickle.loads(bytes_)
            if self.transform:
                img = self.transform(img)
            samples.append((img, label))

        # default collate_fn would handle them
        return samples


if __name__ == '__main__':
    main()