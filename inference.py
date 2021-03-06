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
from datasets import  validation_2cls
from datasets import  val_3cls
from models import *

import numpy as np

try:
    from tensorboard_logger import configure, log_value
except BaseException:
    configure = None


def getModel(**kwargs):
    model = get_model(kwargs['arch'], kwargs['num_classes'])
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    return model


def main():
    # parse arg and start experiment
    global args

    args = arg_parser.parse_args()
    args.config_of_data = config.datasets[args.data]
    args.num_classes = config.datasets[args.data]['num_classes']

    # resume from a checkpoint
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    old_args = checkpoint['args']
    print('Old args:')
    print(old_args)
    # set args based on checkpoint
    for name in arch_resume_names:
        if name in vars(args) and name in vars(old_args):
            setattr(args, name, getattr(old_args, name))

    model = getModel(**vars(args))
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'"
          .format(args.resume))

    cudnn.benchmark = True

    # check if the folder exists
    create_save_folder(args.save, args.force)

    # create dataloader
    if args.data == 'val_2cls':
        loader = validation_2cls.val_2cls()
    elif args.data == 'val_3cls':
        loader = val_3cls.val_3cls()
    else:
        raise NotImplemented

    img_list = loader.get_img_list()

    for i, (img_name, _) in enumerate(img_list):
        img, truth = loader.get_item(i)
        input = img[np.newaxis, :]
        predict = inference(input, model)

        log_path = os.path.join(args.save, 'val_result.csv')

        with open(log_path, 'a') as file:
            content = img_name + ',' + str(predict.data[0]) + ',' + str(truth) + '\n'
            file.write(content)


def inference(input, model):

    model.eval()
    input_var = torch.autograd.Variable(input, volatile=True)

    output = model(input_var)
    _, predict = torch.max(output, dim=1)

    return predict

if __name__ == '__main__':
    main()
