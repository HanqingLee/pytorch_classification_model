from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from utils import AverageMeter, adjust_learning_rate, get_accuracy, recall_value, precision_value, \
    f1_value
import time
import os

from PIL import Image


class Trainer(object):
    def __init__(self, model, criterion=None, optimizer=None, args=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args

    def train(self, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        train_acc = AverageMeter()

        # switch to train mode
        self.model.train()

        lr = adjust_learning_rate(self.optimizer, self.args.lr,
                                  self.args.decay_rate, epoch,
                                  self.args.epochs)  # TODO: add custom
        print('Epoch {:3d} lr = {:.6e}'.format(epoch, lr))

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure recall, precision, f1 value and record loss
            losses.update(loss.data[0], input.size(0))
            train_acc.update(get_accuracy(output, target_var))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Train_Acc {train_acc.avg:.4f}\t'.format(
                    epoch, i + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, train_acc=train_acc))

        print('Epoch: {:3d} Train loss {loss.avg:.4f} '
              'Acc {train_acc.avg:.4f}'
              .format(epoch, loss=losses, train_acc=train_acc))

        print('=======================Epoch %d finished=======================' % epoch)
        return losses.avg, train_acc.avg, lr

    def test(self, val_loader, epoch, silence=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        val_acc = AverageMeter()
        recall_list = {}
        precision_list = {}
        f1_list = {}

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure error and record loss
            losses.update(loss.data[0], input.size(0))
            val_acc.update(get_accuracy(output, target_var))
            recall = recall_value(output, target_var)
            precision = precision_value(output, target_var)
            f1 = f1_value(output, target_var)

            for key in recall:
                recall_val = recall[key]
                precision_val = precision[key]
                f1_val = f1[key]

                if key not in recall_list:
                    recall_list[key] = AverageMeter()
                    precision_list[key] = AverageMeter()
                    f1_list[key] = AverageMeter()

                if recall_val != None:
                    recall_list[key].update(recall_val)
                if precision_val != None:
                    precision_list[key].update(precision_val)
                if f1_val != None:
                    f1_list[key].update(f1_val)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        for key in recall_list:
            recall_list[key] = recall_list[key].avg
            precision_list[key] = precision_list[key].avg
            f1_list[key] = f1_list[key].avg

        if not silence:
            print('Epoch: {:3d} val   '
                  'Loss {loss.val:.4f}\t'
                  'Val_Acc {val_acc.val:.4f}\t'.format(epoch, loss=losses, val_acc=val_acc))

        return losses.avg, val_acc.avg, recall_list, precision_list, f1_list
