#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch import topk
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

import cv2

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
    if args.start_epoch <= 0:
        args.start_epoch = checkpoint['epoch'] + 1
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
    loader = val_3cls.val_3cls()
    img_list = loader.get_img_list()

    class SaveFeatures():
        features = None

        def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

        def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()

        def remove(self): self.hook.remove()

    final_layer = model._modules.get('module').features.conv5_bn_ac

    activated_features = SaveFeatures(final_layer)

    model.eval()


    for i, (img_name, truth) in enumerate(img_list):
        img, truth = loader.get_item(i)
        input = img[np.newaxis, :]

        input_var = torch.autograd.Variable(input, volatile=True)
        output = model(input_var)
        _, predict = torch.max(output, dim=1)

        pred_probabilities = F.softmax(output, dim=1).data.squeeze()
        activated_features.remove()

        topk(pred_probabilities, 1)

        def returnCAM(feature_conv, weight_softmax, class_idx):
            # generate the class activation maps upsample to 256x256
            size_upsample = (256, 256)
            bz, nc, h, w = feature_conv.shape
            output_cam = []
            for idx in class_idx:
                cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                output_cam.append(cv2.resize(cam_img, size_upsample))
            return output_cam

        weight_softmax_params = list(model._modules.get('module').classifier.parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        class_idx = topk(pred_probabilities, 1)[1].int()
        CAMs = returnCAM(activated_features.features, weight_softmax, class_idx)

        img_path = os.path.join(loader.img_dir, img_name)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + img * 0.5

        new_name = ('{}_predict{}_truth{}').format(img_name.split('.')[0], predict.data[0], truth)
        output_path = os.path.join(args.save, new_name + '_CAM.jpg')
        origin_output_path = os.path.join(args.save, new_name + '.jpg')

        cv2.imwrite(origin_output_path, img)
        cv2.imwrite(output_path, result)

def inference(input, model):

    model.eval()
    input_var = torch.autograd.Variable(input, volatile=True)

    output = model(input_var)
    _, predict = torch.max(output, dim=1)

    return predict

if __name__ == '__main__':
    main()
