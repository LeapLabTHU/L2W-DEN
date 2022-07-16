import os
import math
import torch
import shutil
import socket


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None):
    T_total = args.epochs * nBatch
    T_cur = (epoch % args.epochs) * nBatch + batch
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_meta_learning_rate(meta_optimizer, epoch, args, batch=None, nBatch=None):
    T_total = args.epochs * nBatch
    T_cur = (epoch % args.epochs) * nBatch + batch
    meta_lr = 0.5 * args.meta_lr * (1 + math.cos(math.pi * T_cur / T_total))
    for param_group in meta_optimizer.param_groups:
        param_group['lr'] = meta_lr
    return meta_lr
    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write('')

    def log(self, string, isprint=True):
        if isprint:
            print(string)
        with open(self.filename, 'a') as f:
            f.write(str(string)+'\n')


def save_checkpoint(state, args, is_best, filename, result):
    result_filename = os.path.join(args.train_url, 'scores.tsv')
    model_dir = os.path.join(args.train_url, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.train_url, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    args.print_custom("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)

    if is_best:
        torch.save(state, best_filename)

    args.print_custom("=> saved checkpoint '{}'".format(model_filename))
    return


def calc_target_probs(args):
    for p in range(1, 40):
        _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
        probs = torch.exp(torch.log(_p) * torch.arange(1, args.num_exits+1))
        probs /= probs.sum()
        if p == 1:
            probs_list = probs.unsqueeze(0)
        else:
            probs_list = torch.cat((probs_list, probs.unsqueeze(0)), 0)
    
    return probs_list


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    free_port = sock.getsockname()[1]
    return free_port


def get_scores_string(num_exits, prefix, suffix):
    result_str = ""
    for i in range(1, num_exits + 1):
        result_str = result_str + prefix + str(i) + suffix
    return result_str


def get_num_string(data):
    result_str = ""
    for i in range(len(data)):
        result_str = result_str + f"\t{data[i]:.4f}"
    return result_str