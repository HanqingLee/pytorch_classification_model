#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from colorama import Fore
from importlib import import_module

import config
from dataloader import getDataloaders
from utils import save_checkpoint, get_optimizer, create_save_folder, load_pretrained_diff_parameter
from args import arg_parser, arch_resume_names
from models import *

try:
    from tensorboard_logger import configure, log_value
except BaseException:
    configure = None


def getModel(**kwargs):
    model = get_model(kwargs['arch'], kwargs['num_classes'])
    return model


def main():
    # parse arg and start experiment
    global args
    best_acc = 0.
    best_epoch = 0

    args = arg_parser.parse_args()
    args.config_of_data = config.datasets[args.data]
    args.num_classes = config.datasets[args.data]['num_classes']

    # limit the gpu id to use
    # WARNING: This assignment should be down at the beginning, in case of different assignment for different parts
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpu_id = []
    for i in range(len(args.gpu_id.split(','))):
        gpu_id.append(i)

    if configure is None:
        args.tensorboard = False
        print(Fore.RED +
              'WARNING: you don\'t have tesnorboard_logger installed' +
              Fore.RESET)

    # optionally resume from a checkpoint
    if args.resume:
        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            old_args = checkpoint['args']
            print('Old args:')
            print(old_args)
            # set args based on checkpoint
            if args.start_epoch <= 0:
                args.start_epoch = checkpoint['epoch'] + 1
            best_epoch = args.start_epoch - 1
            print('Epoch recovered:%d' % checkpoint['epoch'])
            best_acc = checkpoint['best_acc']
            for name in arch_resume_names:
                if name in vars(args) and name in vars(old_args):
                    setattr(args, name, getattr(old_args, name))

            model = getModel(**vars(args))

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print(
                "=> no checkpoint found at '{}'".format(
                    Fore.RED +
                    args.resume +
                    Fore.RESET),
                file=sys.stderr)
            return
    elif args.pretrain:
        # create model
        print("=> creating model '{}'".format(args.arch))
        model = getModel(**vars(args))
        model = load_pretrained_diff_parameter(model, args.pretrain)

        print("=> pre-train weights loaded")
    else:
        # create model
        print("=> creating model '{}'".format(args.arch))
        model = getModel(**vars(args))

    model = torch.nn.DataParallel(model, device_ids=gpu_id).cuda()

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = get_optimizer(model, args)

    # set random seed
    torch.manual_seed(args.seed)

    Trainer = import_module(args.trainer).Trainer
    trainer = Trainer(model, criterion, optimizer, args)

    # create dataloader
    if args.eval == 'train':
        train_loader, _, _ = getDataloaders(args.data,
                                            splits=('train'), batch_size=args.batch_size)
        trainer.test(train_loader, best_epoch)
        return
    elif args.eval == 'val':
        _, val_loader, _ = getDataloaders(args.data,
                                          splits=('val'), batch_size=args.batch_size)
        trainer.test(val_loader, best_epoch)
        return
    elif args.eval == 'test':
        _, _, test_loader = getDataloaders(args.data, splits=('test'), batch_size=args.batch_size)
        trainer.test(test_loader, best_epoch)
        return
    else:
        train_loader, val_loader, _ = getDataloaders(args.data,
                                                     splits=('train', 'val'),
                                                     batch_size=args.batch_size)

    # check if the folder exists
    create_save_folder(args.save, args.force)

    # set up logging
    global log_print, f_log
    f_log = open(os.path.join(args.save, 'log.txt'), 'w')

    def log_print(*args):
        print(*args)
        print(*args, file=f_log)

    log_print('args:')
    log_print(args)
    # print('model:', file=f_log)
    # print(model, file=f_log)
    f_log.flush()
    log_print('# of params:',
              str(sum([p.numel() for p in model.parameters()])))

    torch.save(args, os.path.join(args.save, 'args.pth'))
    if args.tensorboard:
        configure(args.save, flush_secs=5)

    for epoch in range(args.start_epoch, args.epochs + 1):

        # train for one epoch
        train_loss, train_acc, lr = trainer.train(
            train_loader, epoch)

        if args.tensorboard:
            log_value('lr', lr, epoch)
            log_value('train_loss', train_loss, epoch)
            log_value('train_acc', train_acc, epoch)

        # evaluate on validation set
        val_loss, val_acc, recall, precision, f1, acc = trainer.test(val_loader, epoch, silence=True)

        if args.tensorboard:
            log_value('val_loss', val_loss, epoch)
            log_value('val_acc', val_acc, epoch)
            # log recall, precision and f1 value for every class
            # labels should be sequential natural numbers like 0,1,2....
            for i in range(args.num_classes):
                try:
                    log_value('cls_' + str(i) + '_recall', recall[i], epoch)
                except:
                    log_value('cls_' + str(i) + '_recall', 0, epoch)
                try:
                    log_value('cls_' + str(i) + '_precision', precision[i], epoch)
                except:
                    log_value('cls_' + str(i) + '_precision', 0, epoch)
                try:
                    log_value('cls_' + str(i) + '_f1', f1[i], epoch)
                except:
                    log_value('cls_' + str(i) + '_f1', 0, epoch)
                try:
                    log_value('cls_' + str(i) + 'acc', acc[i], epoch)
                except:
                    log_value('cls_' + str(i) + 'acc', 0, epoch)

        # save scores to a tsv file, rewrite the whole file to prevent
        # accidental deletion
        print(('epoch:{}\tlr:{}\ttrain_loss:{:.4f}\ttrain_acc:{:.4f}\tval_loss:{:.4f}\tval_acc:{:.4f}')
                      .format(epoch, lr, train_loss,train_acc, val_loss, val_acc), file=f_log)
        for i in range(args.num_classes):
            try:
                print(('cls_{}_recall: {:.4f}').format(i, recall[i]), file=f_log)
            except:
                print(('cls_{}_recall: {:.4f}').format(i, 0), file=f_log)
            try:
                print(('cls_{}_precision: {:.4f}').format(i, precision[i]), file=f_log)
            except:
                print(('cls_{}_precision: {:.4f}').format(i, 0), file=f_log)
            try:
                print(('cls_{}_f1: {:.4f}').format(i, f1[i]), file=f_log)
            except:
                print(('cls_{}_f1: {:.4f}').format(i, 0), file=f_log)
            try:
                print(('cls_{}_acc: {:.4f}').format(i, acc[i]), file=f_log)
            except:
                print(('cls_{}_acc: {:.4f}').format(i, 0), file=f_log)
        f_log.flush()

        # remember best err@1 and save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
            print(Fore.GREEN + 'Best var_acc {}'.format(best_acc) + Fore.RESET, file=f_log)
        f_log.flush()

        dict = {
            'args': args,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_acc': best_acc,
        }
        # state_dict: model.state_dict() will add "module" layer in front of every model. The reading of this kind of
        # checkpoint requires to initialize model with DataParallel before resuming.
        save_checkpoint(dict, is_best, args.save, filename='checkpoint_' + str(epoch) + '.pth.tar')
        if not is_best and epoch - best_epoch >= args.patience > 0:
            break
    print('Best best_acc: {:.4f} at epoch {}'.format(best_acc, best_epoch), file=f_log)

if __name__ == '__main__':
    main()
