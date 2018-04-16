import sys
import time
import os
import shutil
import torch

from colorama import Fore


def create_save_folder(save_path, force=False, ignore_patterns=[]):
    if os.path.exists(save_path):
        print(Fore.RED + save_path + Fore.RESET + ' already exists!')
        # if not force:
        #     ans = input('Do you want to overwrite it? [y/N]:')
            # if ans not in ('y', 'Y', 'yes', 'Yes'):
            #     os.exit(1)
        from getpass import getuser
        tmp_path = '/tmp/{}-experiments/{}_{}'.format(getuser(),
                                                      os.path.basename(save_path),
                                                      time.time())
        print('move existing {} to {}'.format(save_path, Fore.RED
                                              + tmp_path + Fore.RESET))
        shutil.copytree(save_path, tmp_path)
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print('create folder: ' + Fore.GREEN + save_path + Fore.RESET)

    # copy code to save folder
    if save_path.find('debug') < 0:
        shutil.copytree('.', os.path.join(save_path, 'src'), symlinks=True,
                        ignore=shutil.ignore_patterns('*.pyc', '__pycache__',
                                                      '*.path.tar', '*.pth',
                                                      '*.ipynb', '.*', 'data',
                                                      'save', 'save_backup',
                                                      save_path,
                                                      *ignore_patterns))


def load_pretrained_diff_parameter(model, model_path):
    # model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    # diff = {k: v for k, v in model_dict.items() if \
    #         k in pretrained_dict and pretrained_dict[k].size() != v.size()}
    # pretrained_dict.update(diff)
    model.load_state_dict(pretrained_dict)
    return model

def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch >= num_epochs * 0.75:
        lr *= decay_rate ** 2
    elif epoch >= num_epochs * 0.5:
        lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


def get_optimizer(model, args):
    import sys
    sys.path.insert(0, '/home/ubuntu/skin_demo/RoP/AMSGrad/')

    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), args.lr,
                                beta=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)

    elif args.optimizer == 'adam_1':
        import adam
        optimizer = adam.Adam([{'params': model.parameters(), 'lr': args.lr}
                               ], lr=args.lr, weight_decay=0.0001, amsgrad=True)
        return optimizer

    elif args.optimizer == 'adam_2':
        import adam_amsgrad
        optimizer = adam_amsgrad.Adam([{'params': model.parameters(), 'lr': args.lr}
                               ], lr=args.lr, weight_decay=0.0001, amsgrad=True)
        return optimizer

    elif args.optimizer == 'adam_3':
        import adamw
        optimizer = adamw.Adam([{'params': model.parameters(), 'lr': args.lr}
                               ], lr=args.lr, weight_decay=0.0001)
        return optimizer

    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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
        self.avg = float(self.sum) / float(self.count)


def error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res


def get_accuracy(output, target):
    _, predict = torch.max(output, dim=1)
    train_correct = (predict.data == target.data).sum()
    train_acc = float(float(train_correct) / float(len(predict)))
    return train_acc


def get_TF_table(output, target):
    labels = dict()
    test = {'a':0, 'b':1}
    for i in range(len(target)):
        if target.data[i] not in labels:
            labels[target.data[i]] = [0, 0, 0, 0]  # TP_num, FP_num, FN_num, TN_num

    _, predict = torch.max(output, dim=1)
    for key in labels:
        TF_list = labels[key]
        for i in range(len(predict)):
            predict_single = int(predict.data[i])
            truth = int(target.data[i])
            if predict_single == int(key) and truth == int(key):
                TF_list[0] += 1
            elif predict_single == int(key) and truth != int(key):
                TF_list[1] += 1
            elif predict_single != int(key) and truth == int(key):
                TF_list[2] += 1
            elif predict_single != int(key) and truth != int(key):
                TF_list[3] += 1
            else:
                print(
                            'Exception found when calculating TF table between:\npredict: %s\ntruth: %s'
                            % (str(predict), str(truth)))
                raise Exception
    return labels


def recall_value(output, target):
    labels = get_TF_table(output, target)
    recall_list = {}
    for key in labels:
        try:
            TF_list = labels[key]
            TP = float(TF_list[0])
            FN = float(TF_list[2])
            recall_list[key] = float(TP / (TP + FN))
        except:
            recall_list[key] = None
    return recall_list


def precision_value(output, target):
    labels = get_TF_table(output, target)
    precision_list = {}
    for key in labels:
        try:
            TF_list = labels[key]
            TP = float(TF_list[0])
            FP = float(TF_list[1])
            precision_list[key] = float(TP / (TP + FP))
        except:
            precision_list[key] = None
    return precision_list


def f1_value(output, target):
    labels = get_TF_table(output, target)
    f1_list = {}
    for key in labels:
        try:
            TF_list = labels[key]
            TP = float(TF_list[0])
            FN = float(TF_list[2])
            FP = float(TF_list[1])
            f1_list[key] = float(2 * TP / (2 * TP + FN + FP))
        except:
            f1_list[key] = None
    return f1_list
