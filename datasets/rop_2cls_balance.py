from __future__ import print_function
from PIL import Image

import torch.utils.data as data
from torchvision import transforms

class rop_2cls_balance(data.Dataset):

    def __init__(self, category, transform=None, target_transform=None):
        self.img_dir = '/home/ubuntu/skin_demo/RoP/data/ROP_All_CLANE_enhanced_balanced/'
        self.train_dataTXT = "/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_2cls_train_balanced.txt"
        self.val_dataTXT = "/home/hq/workspace/RoP/label/ROP_Phase/jsonAll_phases_3cls_only12_val.txt"
        self.test_dataTXT = ''

        sizex, sizey = 320, 240

        self.transform = transforms.Compose([
            transforms.Resize([sizex, sizey]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if category == 'train':
            txt_name = self.train_dataTXT
        elif category == 'val':
            txt_name = self.val_dataTXT
        else:
            raise NotImplemented

        with open(txt_name) as f:
            data = f.readlines()
        img_list = []
        for line in data:
            words = line.strip().split(',')
            img_list.append((words[0], int(words[1])))

        self.img_list = img_list
        if transform != None:
            self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, label = self.img_list[index]
        img = Image.open(self.img_dir + img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_list)
